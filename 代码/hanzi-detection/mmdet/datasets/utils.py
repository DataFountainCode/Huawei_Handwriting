import copy
from collections import Sequence

import cv2 
import mmcv
from mmcv.runner import obj_from_dict
import torch

import matplotlib.pyplot as plt
import numpy as np
from .concat_dataset import ConcatDataset
from .repeat_dataset import RepeatDataset
from .. import datasets


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            ratio=max(img_scales[0])/min(img_scales[0])
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            """
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            """
            short_edge = int(long_edge/ratio)    
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError('Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]
    return img_scale

def crop(img, img_shape, crop_ratio, gt_bboxes, gt_labels, size_divisor):
    img = copy.deepcopy(img)
    img_shape = copy.deepcopy(img_shape)
    gt_bboxes = copy.deepcopy(gt_bboxes)
    gt_labels = copy.deepcopy(gt_labels)
    img = img.transpose(1, 2, 0)
    height, width, _ = img_shape
    left = np.random.randint(width - int(width * crop_ratio))
    top = np.random.randint(height - int(height * crop_ratio))
    img = img[top:top+int(height * crop_ratio), left:left+int(width * crop_ratio), :]  # crop image
    img_shape = img.shape

    # adjust gt bboxes to fit croped image.
    i = 0
    while(i< gt_bboxes.shape[0]):
        if (gt_bboxes[i, 0]>left+width)or(gt_bboxes[i, 2]<left)or(gt_bboxes[i, 1]>top+height)or(gt_bboxes[i, 3]<top):
            # gt bboxes out of range of croped image
            gt_bboxes = np.delete(gt_bboxes, i, axis=0)
            gt_labels = np.delete(gt_labels, i)
        else:
            # gt bboxes in croped image or over range of some parts
            if gt_bboxes[i, 0] < left :
                gt_bboxes[i, 0] = left
            if gt_bboxes[i, 2] > left + width - 1:
                gt_bboxes[i, 2] = left + width - 1
            if gt_bboxes[i, 1] < top :
                gt_bboxes[i, 1] = top 
            if gt_bboxes[i, 3] > top + height - 1:
                gt_bboxes[i, 3] = top + height - 1 
            i = i+1    
    gt_bboxes[:, [0, 2]]=gt_bboxes[:, [0, 2]]-left
    gt_bboxes[:, [1, 3]]=gt_bboxes[:, [1, 3]]-top  
    img = mmcv.impad_to_multiple(img, size_divisor) 
    pad_shape = img.shape
    img = img.transpose(2, 0, 1)
    return img, img_shape, pad_shape, gt_bboxes, gt_labels  
                        
def get_stretch_ratio(stretch_ratio, stretch_mode='value'):
    if stretch_mode=='value':
        return stretch_ratio[np.random.randint(len(stretch_ratio))]
    elif stretch_mode=='range' :
        assert len(stretch_ratio)==2
        width_stretch=[s[0] for s in stretch_ratio]
        height_stretch=[s[1] for s in stretch_ratio]    
        return (np.random.uniform(min(width_stretch),max(width_stretch)), np.random.uniform(min(height_stretch),max(height_stretch)))

def stretch(img, img_shape, gt_bboxes, gt_bboxes_8_coo, stretch_ratio, size_divisor):
    img = img.transpose(1,2,0)
    img = img[:img_shape[0], :img_shape[1], :]  # crop from padded image

    height = int(img_shape[0]*stretch_ratio[1])
    width = int(img_shape[1]*stretch_ratio[0])
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC) 
    gt_bboxes[:, [0, 2]] = gt_bboxes[:, [0, 2]]*(width/img_shape[1])
    gt_bboxes[:, [1, 3]] = gt_bboxes[:, [1, 3]]*(height/img_shape[0])
    gt_bboxes_8_coo[:,0::2] = gt_bboxes_8_coo[:,0::2]*(width/img_shape[1])
    gt_bboxes_8_coo[:,1::2] = gt_bboxes_8_coo[:,1::2]*(width/img_shape[1])
    
    img_shape=(height, width, 3)
    img = mmcv.impad_to_multiple(img, size_divisor)      
    pad_shape = img.shape
    img = img.transpose(2, 0, 1)
    return img, img_shape, pad_shape, gt_bboxes, gt_bboxes_8_coo
                   
def show_ann(coco, img, ann_info):
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    coco.showAnns(ann_info)
    plt.show()


def get_dataset(data_cfg):
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset']), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if 'proposal_file' in data_cfg.keys():
        if isinstance(data_cfg['proposal_file'], (list, tuple)):
            proposal_files = data_cfg['proposal_file']
        else:
            proposal_files = [data_cfg['proposal_file']]
    else:
        proposal_files = [None] * num_dset
    assert len(proposal_files) == num_dset

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        
        data_info = copy.deepcopy(data_cfg)
        print(data_info)
        data_info['ann_file'] = ann_files[i]
        data_info['proposal_file'] = proposal_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset
