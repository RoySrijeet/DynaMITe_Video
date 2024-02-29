# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import datetime
import logging
logging.basicConfig(level=logging.INFO)
import os
import time
from contextlib import ExitStack, contextmanager
import copy
import numpy as np
import torch
import random
import torchvision
from collections import defaultdict

from detectron2.utils.colormap import colormap
from detectron2.utils.comm import get_world_size        # utils.comm -> primitives for multi-gpu communication
from detectron2.utils.logger import log_every_n_seconds
from torch import nn
# from ..clicker import Clicker
from ..utils.clicker import Clicker
from ..utils.predictor import Predictor

def evaluate(
    model, data_loader, iou_threshold = 0.85, max_interactions = 10, sampling_strategy=1,
    eval_strategy = "worst", seed_id = 0, vis_path = None, video_mode=False
):
    """
    Run model on the data_loader and return a dict, later used to calculate
    all the metrics for multi-instance inteactive segmentation such as NCI,
    NFO, NFI, and Avg IoU.
    The model will be used in eval mode.

    Arguments:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        iou_threshold: float
            Desired IoU value for each object mask
        max_interactions: int
            Maxinum number of interactions per object
        sampling_strategy: int
            Strategy to avaoid regions while sampling next clicks
            0: new click sampling avoids all the previously sampled click locations
            1: new click sampling avoids all locations upto radius 5 around all
               the previously sampled click locations
        eval_strategy: str
            Click sampling strategy during refinement
        seed_id: int
            Used to generate fixed seed during evaluation
        vis_path: str
            Path to save visualization of masks with clicks during evaluation
        video_mode: bool
            If set to True, the input images are frames of a video sequence

    Returns:
        Dict with following keys:
            'total_num_instances': total number of instances in the dataset
            'total_num_interactions': total number of interactions/clicks sampled 
            'total_compute_time_str': total compute time for evaluating the dataset
            'iou_threshold': iou_threshold
            'num_interactions_per_image': a dict with keys as image ids and values 
             as total number of interactions per image
            'final_iou_per_object': a dict with keys as image ids and values as
             list of ious of all objects after final interaction
    """
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))                       # 1999 (davis_2017_val)
    logger.info(f"Using {eval_strategy} evaluation strategy with random seed {seed_id}")

    total = len(data_loader)  # inference data loader must have a fixed length
   
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0 
    total_compute_time = 0
    total_eval_time = 0
    
    # VID
    sequence_name = None
    
    with ExitStack() as stack:                                  # managing multiple context managers

        print(f'[INFO] Calling context managers...')
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))       # (context manager) set the model temporarily to .eval()
        stack.enter_context(torch.no_grad())                    # (context manager) disable gradient calculation

        print(f'[INFO] Initializing variables...')
        total_num_instances = 0         # in the dataset                               
        total_num_interactions = 0      # that were sampled
        
        final_iou_per_object = defaultdict(list)        # will store IoUs for all objects (in a list), for each image (image-id as key)
        num_interactions_per_image = {}                 # key: image-id, value: #interactions

        random.seed(123456+seed_id)
        start_data_time = time.perf_counter()

        first_frame_clicks = None
        print(f'[INFO] Starting iteration through the Data Loader...')
        # iterate through the data_loader, one image at a time
        for idx, inputs in enumerate(data_loader):
            
            print(f'[INFO] Index {idx}: Frame {inputs[0]["file_name"]}')
            #print(f'[INFO] Inputs info: type: {type(inputs)}, length: {len(inputs)}')       # list, len 1 (davis_2017_val)
            #print(f'[INFO] Inputs first element... type: {type(inputs[0])}')               # dict
            #print(f'[INFO] Inputs first element... type: {inputs[0].keys()}')                
                    # dict_keys([
                    # 'file_name' - full path to image, 
                    # 'height' - at orig res (H'),     
                    # 'width' - at orig res (W'), 
                    # 'image_id' - <sequence_name><filename>, e.g., (bike-packing000000), 
                    # 'image' - after transformation, torch.Tensor - CxHxW, 
                    # 'padding_mask' - with same res as transformed image, torch.Tensor - HxW, 
                    # 'semantic_map' - ground truth annotation (all instances) at orig res - H'xW', 
                    # 'orig_fg_click_coords' - at image res, instance-wise list of [x,y,time_step] lists (time_step - interaction count per image) 
                    # 'fg_click_coords' - at res after image transformations, instance-wise list of [x,y,time_step] lists, 
                    # 'bg_click_coords' - at res after image transformations, instance-wise list of [x,y,time_step] lists,
                    # 'num_clicks_per_object' - list of counters, one for each instance in the image, 
                    # 'instances - d2.structures.Instances object with image metadata and instance-wise mask info (at orig res)'
                #])

            # VID
            if video_mode:
                new_seq = False
                curr_seq_name = inputs[0]["file_name"].split('/')[-2]
                if sequence_name is None or sequence_name!=curr_seq_name:
                    sequence_name = curr_seq_name
                    print(f'[INFO] New sequence found: {sequence_name}')
                    new_seq = True
            
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            
            # initialize Clicker, load ground truth info
            if video_mode:
                if new_seq:
                    #print(f'[INFO] Creating Clicker for first frame of the new sequence...')
                    clicker = Clicker(inputs, new_seq, sampling_strategy)
                    first_frame_clicks = [copy.deepcopy(clicker.fg_coords), 
                                          copy.deepcopy(clicker.bg_coords),
                                          copy.deepcopy(clicker.fg_orig_coords), 
                                          copy.deepcopy(clicker.bg_orig_coords)]
                else:
                    #print(f'[INFO] Creating Clicker for {idx+1} frame...')
                    clicker = Clicker(inputs, new_seq, sampling_strategy, first_frame_clicks=first_frame_clicks)
            else:
                clicker = Clicker(inputs, True, sampling_strategy)  
            # initialize Predictor  (only set model, rest of the properties to be set after the first prediction call)
            predictor = Predictor(model)

            if vis_path:
                clicker.save_visualization(vis_path, ious=[0], num_interactions=0)  # num_interactions==0: ground truth masks
            
            # #instances derived from inputs[0]['instances'].gt_masks (#channels)
            num_instances = clicker.num_instances
            total_num_instances+=num_instances      # counter for whole dataset

            # we start with atleast one interaction per instance (center of each instance, on the ground truth mask)
            total_num_interactions+=(num_instances)

            num_interactions = num_instances                # one interaction per instance
            num_clicks_per_object = [1]*(num_instances+1)   # +1 for background
            num_clicks_per_object[-1] = 0                   # no interaction for bg yet, so reset

            # budget defined per instance (max_interactions=10 per instance)
            max_iters_for_image = max_interactions * num_instances

            # first call - from the clicker object, take the input sample, and max_timestamps (?) to make predictions
            # first call also populates many Predictor attributes (see inference.utils.predictor.py)
            pred_masks = predictor.get_prediction(clicker)      # at orig (pre-transfn) image res (num_inst xHxW)
            clicker.set_pred_masks(pred_masks)                  # clicker.pred_masks 
            ious = clicker.compute_iou()                        # compute iou (one score per channel==instance)
            #print(f'[INFO] IoU after first call: length: {len(ious)},ious: {ious}') # list of len==num_instances

            if vis_path:
                clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions)  # viz pred mask

            point_sampled = True

            random_indexes = list(range(len(ious)))

            #interative refinement loop
            while (num_interactions<max_iters_for_image):       # if not over-budget
                if all(iou >= iou_threshold for iou in ious):   # if mask quality met for all instances
                    break

                index_clicked = [False]*(num_instances+1)   # redundant - probably
                if eval_strategy == "worst":
                    # returns a list of indices that sorts ious list from lowest to highest
                    indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=False).indices
                elif eval_strategy == "best":
                    # returns a list of indices that sorts ious list from highest to lowest
                    indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=True).indices
                elif eval_strategy == "random":
                    random.shuffle(random_indexes)
                    indexes = random_indexes
                else:
                    assert eval_strategy in ["worst", "best", "random"]

                point_sampled = False
                for i in indexes:
                    # sample click on the first instance that has iou below threshold
                    if ious[i]<iou_threshold: 
                        obj_index = clicker.get_next_click(refine_obj_index=i, time_step=num_interactions)  #num_interactions - counter over image
                        total_num_interactions+=1   # for dataset
                        
                        index_clicked[obj_index] = True
                        num_clicks_per_object[i]+=1         # update counter for instances
                        point_sampled = True
                        break
                if point_sampled:
                    num_interactions+=1     # for image
          
                    pred_masks = predictor.get_prediction(clicker)
                    clicker.set_pred_masks(pred_masks)
                    ious = clicker.compute_iou()
                    
                    if vis_path:
                        clicker.save_visualization(vis_path, ious=ious, num_interactions=num_interactions)
                    # final_iou_per_object[f"{inputs[0]['image_id']}_{idx}"].append(ious)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
           
            final_iou_per_object[f"{inputs[0]['image_id']}_{idx}"].append(ious)
            num_interactions_per_image[f"{inputs[0]['image_id']}_{idx}"] = num_interactions
            
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        # f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"Total instances: {total_num_instances}. "
                        f"Average interactions:{(total_num_interactions/total_num_instances):.2f}. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        ),
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = {'total_num_instances': [total_num_instances],
                'total_num_interactions': [total_num_interactions],
                'total_compute_time_str': total_compute_time_str,
                'iou_threshold': iou_threshold,
                'final_iou_per_object': [final_iou_per_object],
                'num_interactions_per_image': [num_interactions_per_image],
    }

    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
