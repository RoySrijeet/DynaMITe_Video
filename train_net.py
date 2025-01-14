#Adapted by Amit Rana from: https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py


import csv

import numpy as np

try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging

from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    launch,
)
from dynamite.utils.misc import default_argument_parser

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from dynamite import (
    COCOLVISDatasetMapper, EvaluationDatasetMapper
)

from dynamite import (
    add_maskformer2_config,
    add_hrnet_config
)

from dynamite.inference.utils.eval_utils import log_single_instance, log_multi_instance

#import wandb
#wandb.init(entity='thesis-roy', project='dynamite_video', name='dynamite_train', sync_tensorboard=True)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to Mask2Former.
    """

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        mapper = EvaluationDatasetMapper(cfg,False,dataset_name)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)        # d2 call
        
    @classmethod
    def build_train_loader(cls, cfg):
        datset_mapper_name = cfg.INPUT.DATASET_MAPPER_NAME
        if datset_mapper_name == "coco_lvis":
            mapper = COCOLVISDatasetMapper(cfg,True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        
        """
        Method is called after every evaluation Checkpoint iteration.
        You can evaluate on any dataset and log the results/metrics 
        for debugging and performance measure puposes.
        """
        cls.interactive_evaluation(cfg,model)
        return {}

    @classmethod
    def interactive_evaluation(cls, cfg, model, args=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """
        print('[INFO] Interactive Evaluation started...')
        if not args:
            return 

        logger = logging.getLogger(__name__)

        if args and args.eval_only:
            eval_datasets = args.eval_datasets      # dataset to run evaluation on
            vis_path = args.vis_path                
            eval_strategy = args.eval_strategy      # "random", "best", "worst", "max_dt", "wlb", "round_robin"
            seed_id = args.seed_id
            iou_threshold = args.iou_threshold
            max_interactions = args.max_interactions
        
        # assert iou_threshold in [0.80, 0.85, 0.90, 0.95, 1.00]
        assert iou_threshold>=0.80

        print(f'[INFO] Evaluation datasets: {eval_datasets}')
        print(f'[INFO] Evaluation strategy: {eval_strategy}')
        print(f'[INFO] IoU Threshold: {iou_threshold}')
        print(f'[INFO] Max interaction limit: {max_interactions}')

        for dataset_name in eval_datasets:

            if dataset_name in ["GrabCut", "Berkeley", "davis_single_inst", "coco_Mval", 'sbd_single_inst']:
                from dynamite.inference.single_instance.single_instance_inference import get_avg_noc

                # from dynamite.inference.single_instance.sam_inference import get_avg_noc
                data_loader = cls.build_test_loader(cfg, dataset_name)
                
                results_i = get_avg_noc(model, data_loader, iou_threshold = iou_threshold,
                                        sampling_strategy=1, max_interactions=max_interactions,
                                        vis_path=vis_path
                                        )
                results_i = comm.gather(results_i, dst=0)  # [res1:dict, res2:dict,...]
                if comm.is_main_process():
                    # sum the values with same keys
                    assert len(results_i)>0
                    res_gathered = results_i[0]
                    results_i.pop(0)
                    for _d in results_i:
                        for k in _d.keys():
                            res_gathered[k] += _d[k]
                    log_single_instance(res_gathered, max_interactions=max_interactions, 
                                        dataset_name=dataset_name, iou_threshold=iou_threshold)
            
            # multi-instance eval
            elif dataset_name in ["davis_2017_val","sbd_multi_insts","coco_2017_val"]:
                print(f'[INFO] Initiating Multi-Instance Evaluation on {eval_datasets}...')
                
                if eval_strategy in ["random", "best", "worst"]:
                    from dynamite.inference.multi_instance.random_best_worst import evaluate
                elif eval_strategy == "max_dt":
                    from dynamite.inference.multi_instance.max_dt import evaluate
                elif eval_strategy == "wlb":
                    from dynamite.inference.multi_instance.wlb import evaluate
                elif eval_strategy == "round_robin":
                    from dynamite.inference.multi_instance.round_robin import evaluate
                print(f'[INFO] Loaded Evaluation routine following {eval_strategy} evaluation strategy!')
                
                print(f'[INFO] Loading test data loader from {dataset_name}...')
                data_loader = cls.build_test_loader(cfg, dataset_name)      # creates evaluation dataset mapper and calls d2 test_loader
                print(f'[INFO] Data loader  preparation complete!')
                print(f'[INFO] Data loader info:')
                print(f'[INFO] type: {type(data_loader)}')
                print(f'[INFO] length: {len(data_loader)}')
                
                if dataset_name=="davis_2017_val":
                    video_mode = True
                else:
                    video_mode = False
                print(f'[INFO] Starting evaluation...')
                results_i = evaluate(model, data_loader, iou_threshold = iou_threshold,
                                    max_interactions = max_interactions,
                                    eval_strategy = eval_strategy, seed_id=seed_id,
                                    vis_path=vis_path,video_mode=video_mode)
                print(f'[INFO] Evaluation complete!')
                
                results_i = comm.gather(results_i, dst=0)  # [res1:dict, res2:dict,...]
                if comm.is_main_process():
                    # sum the values with same keys
                    assert len(results_i) > 0
                    res_gathered = results_i[0]
                    results_i.pop(0)
                    for _d in results_i:
                        for k in _d.keys():
                            res_gathered[k] += _d[k]
                    log_multi_instance(res_gathered, max_interactions=max_interactions,
                                    dataset_name=dataset_name, iou_threshold=iou_threshold)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    print('[INFO] Setting up DynaMITE...')
    cfg = get_cfg()                             # cfg object
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)                 
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)       # path to config file
    cfg.merge_from_list(args.opts)
    cfg.freeze()                                # make cfg (and children) immutable
    default_setup(cfg, args)                    # D2 call
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dynamite")
    return cfg


def main(args):
    
    cfg = setup(args)       # create configs 
    print('[INFO] Setup complete!')

    # for evaluation
    if args.eval_only:
        print('[INFO] DynaMITE Evaluation!')
        print('[INFO] Building model...')
        model = Trainer.build_model(cfg)                                                # load model (torch.nn.Module)
        print('[INFO] Loading model weights...')                                        
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(           # d2 checkpoint load
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        print('[INFO] Model loaded!')
        # res = Trainer.test(cfg, model)
        res = Trainer.interactive_evaluation(cfg,model, args)                           # evaluation

        return res

    # for training
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(                                                                             # d2 launch
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
