import numpy as np

from .custom import CustomDataset


class KittiDataset(CustomDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,         # padding image size to the multiple of size_divisor, only pad right and bottom.
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 scale_mode='range',
                 with_crop=False,
                 crop_ratio=1,          # ratio: croped image size / origin image size
                 with_stretch=False,    # scale image height and width size ratio
                 stretch_ratio=None,
                 stretch_mode='value',                 
                 test_mode=False,
                 extra_aug=None,
                 resize_keep_ratio=True,
                 times=10):
        super(KittiDataset, self).__init__(
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor,
                 proposal_file,
                 num_max_proposals,
                 flip_ratio,
                 with_mask,
                 with_crowd,
                 with_label,
                 scale_mode,
                 with_crop,
                 crop_ratio,
                 with_stretch,
                 stretch_ratio,
                 stretch_mode,                 
                 test_mode
         
                 )