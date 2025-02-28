# Modified by Amit Rana from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch
import torchvision
import pycocotools.mask as mask_util
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, Boxes, BoxMode
from detectron2.structures.masks import PolygonMasks

# from dynamite.data.dataset_mappers.utils.datamapper_utils import convert_coco_poly_to_mask,  build_transform_gen

from dynamite.data.dataset_mappers.utils import convert_coco_poly_to_mask, build_transform_gen
from dynamite.inference.utils.eval_utils import get_gt_clicks_coords_eval

__all__ = ["EvaluationDatasetMapper"]

# This is specifically designed for the COCO dataset.
class EvaluationDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DynaMITe.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    5. Prepare a list of foreground clicks (one click per object) for all the objects in the image
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        dataset_name=None,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[EvaluationDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )
        #print('[INFO] EvaluationDatasetMapper called...')
        #print(f'[INFO] tfm gens info: type {type(tfm_gens)}')   # list
        #print(f'[INFO] tfm gens info: length {len(tfm_gens)}')  # 1
        #print(f'[INFO] tfm gens info: element type {type(tfm_gens[0])}')  # detectron2.data.transforms.augmentation_impl.ResizeShortestEdge

        self.img_format = image_format
        self.is_train = is_train
        self.dataset_name = dataset_name
    
    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # reads the file as a PIL.Image, then converts it to a np.ndarray
        # supported types: modes supported in PIL, or "BGR" or "YUV-BT.601"
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)     

        # check image resolution (height and width) with specifications mentioned in dataset_dict,
        # if dataset_dict is missing 'height' and 'width' entries, populate them with image res
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        orig_image_shape = image.shape[:2]
        padding_mask = np.ones(image.shape[:2])                                     # initialize padding mask

        # apply transformations on the image
        # additionally returns the "deterministic" transformations (fvcore.transforms.transform.TransformList)
        # github: d2.data.transforms.augmentation.py
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)            
        
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)                  # apply same transformations as image to the mask
        padding_mask = ~ padding_mask.astype(bool)                                  

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))     # re-arrange image to C,H,W 
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        # annotations - a list of dicts, one for each instance in the image
        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:        # for each instance, there's a dict
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)                 # remove 'keypoints' property (visibility), if exists

            annos = [
                original_res_annotations(obj, orig_image_shape)
                for obj in dataset_dict.pop("annotations")          # check if an object is labeled as COCO's 'crowd region'
                if obj.get("iscrowd", 0) == 0                       # if not, update annotation properties (bbox and seg mask)
            ]
            # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            if self.dataset_name == "coco_2017_val":        
                instances = utils.annotations_to_instances(annos, orig_image_shape)     # d2 call - convert annos (list[dict]) to Instances object (detectron2.data.detection_utils.annotations_to_instances)
            else:
                instances = utils.annotations_to_instances(annos, orig_image_shape,  mask_format="bitmask")
           
            # if the instance has no ground truth mask, return
            if not hasattr(instances, 'gt_masks'):
                return None
            
            # if ground truth mask exists, get a tight bounding box around the mask
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            
            if len(instances) == 0:
                return None
            # Generate masks from polygon
            h, w = instances.image_size
        
            if hasattr(instances, 'gt_masks'):
                if self.dataset_name == "coco_2017_val":
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                else:
                    gt_masks = instances.gt_masks.tensor

                new_gt_masks, instance_map = get_instance_map(gt_masks)
                dataset_dict['semantic_map'] = instance_map

                new_gt_classes = [0]*new_gt_masks.shape[0]
                new_gt_boxes =  Boxes((np.zeros((new_gt_masks.shape[0],4))))
                
                # create a new Instance object with all the reqd properties
                # this will go into the dataset dictionary of this image
                new_instances = Instances(image_size=image_shape)           # image shape after transformations
                new_instances.set('gt_masks', new_gt_masks)
                new_instances.set('gt_classes', new_gt_classes)
                new_instances.set('gt_boxes', new_gt_boxes) 
               
                ignore_masks = None
                if 'ignore_mask' in dataset_dict:
                    ignore_masks = dataset_dict['ignore_mask'].to(device='cpu', dtype = torch.uint8)

                # orig_fg_coords_list = FG click coordinates
                # fg_coords_list = orig_fg_coords scaled to meet image resolutions
                (num_clicks_per_object, fg_coords_list, orig_fg_coords_list) = get_gt_clicks_coords_eval(new_gt_masks, image_shape, ignore_masks=ignore_masks)
        
                dataset_dict["orig_fg_click_coords"] = orig_fg_coords_list
                dataset_dict["fg_click_coords"] = fg_coords_list
                dataset_dict["bg_click_coords"] = None
                dataset_dict["num_clicks_per_object"] = num_clicks_per_object
                assert len(num_clicks_per_object) == gt_masks.shape[0]
            else:
                return None

            dataset_dict["instances"] = new_instances

        return dataset_dict

def get_instance_map(masks):

    mask_areas = torch.sum(masks, (1,2))
    masks = masks.to(dtype=torch.uint8)
    masks =  masks[sorted(range(len(mask_areas)),key=mask_areas.__getitem__,reverse=True)]

    instance_map = torch.zeros((masks.shape[-2:]), dtype=torch.int16)
    num_objects = masks.shape[0]
    instances_ids = np.arange(1, num_objects + 1)

    for _id, _m in enumerate(masks):
        instance_map[_m == 1] = _id+1
        assert (_m != 0).sum() > 0
    
    new_masks = []
    for _id in instances_ids:
        _m = (instance_map == _id).to(dtype=torch.uint8)
        new_masks.append(_m)
    
    if not len(new_masks):
        return None, None
    new_masks = torch.stack(new_masks,dim=0)
    assert num_objects == new_masks.shape[0]
    return new_masks, instance_map

def original_res_annotations(
    annotation, image_size
):
    """
        annotation: a dictionary with annotations for an instance in the image
        image_size: original resolution of the image (before applying transformations)
    """
    # convert bounding box (list of 4 numbers) format to BoxMode.XYXY_ABS
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)   # (box, from_mode, to_mode)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])      # element-wise minimum - to ensure bbox is within image res
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    # segmentation mask of the instance - convert input data to a list of masks
    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons (one for each connected component of the object)
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in polygons
            ]
        elif isinstance(segm, dict):
            # RLE (COCO format)
            mask = mask_util.decode(segm)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask

    return annotation