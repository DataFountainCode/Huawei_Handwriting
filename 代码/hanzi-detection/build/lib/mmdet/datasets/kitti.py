import numpy as np

from .custom import CustomDataset


class KittiDataset(CustomDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 num_max_proposals=1000,
                 flip_ratio=0,
                 with_mask=True,
                 with_crowd=True,
                 with_label=True,
                 test_mode=False):
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
                 test_mode)
