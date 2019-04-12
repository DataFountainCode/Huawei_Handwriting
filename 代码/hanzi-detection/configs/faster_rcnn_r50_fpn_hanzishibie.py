# model settings
model = dict(
    type='FasterRCNN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[20, 10, 4, 1, 1/1.2],   # 0.05 0.1 0.25 0.5 1.0 || 20, 10, 4, 1, 1/1.2
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True,
        with_8_coo=True),                                                                              ###########################
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]
         ),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,                                                                           ################################
        roi_feat_size=7,
        num_classes=1+1,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        level='single',                                                                                     #'single'单层feature，'two_side'上下层,'all'全部4层
        merge_mode='avr_pooling',                                                                                #'concat'或'avr_pooling'
        num_level=4),
    with_8_coo = True                                                                                              #################### 
    )
# label vocab
label_vocab = ['background', 'Hanzi']

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False,
        with_gt_bboxes=False,                                                             #################################
        gt_bboxes_scale=[0.75, 1, 1.5]),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        pos_weight=-1,
        debug=False,
        with_hard_example_mining=False,                                                  ################################
        with_reg=True
        ))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7, #0.7
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, 
        nms=dict(
            type='nms', 
            iou_thr=0.7  #0.5
        ), 
        max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'KittiDataset'
dataset_root = '/home/chenriquan/Datasets/hanzishibie/'
ann_root = dataset_root + 'train_&_val_pkl/'                                                ############################
data_root = dataset_root 
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=5,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_root + 'train_8_coo.pkl',
        img_prefix=data_root + 'traindataset/train_image/',
        img_scale=(600, 960),   # 600 / 960                                       #[(1660, 492), (960, 280)]
        img_norm_cfg=img_norm_cfg,
        scale_mode='value',                                                           #'value'为离散的选择img_scale, 'range'在[(x1,y1),(x2,y2)]中，宽在(x1,x2)中随机选取，宽高比为x1/y1
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        with_crop=False,                                                              ################################
        crop_ratio=224/256,
        with_stretch=False,                                                             ########################
        stretch_ratio=[(0.5, 0.5), (2, 2)],                                  ########################
        stretch_mode='range'                                                      #'value'为离散的随机选择stretch_ratio中的一个，'range'为在[(x1,y1),(x2,y2)]中，宽在(x1,x2),高在(y1,y2)中随机选取，对宽高分别做拉伸
        ),
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'val_8_coo.pkl',
        img_prefix=data_root + 'traindataset/train_image/',             #####################
        img_scale=(600, 960),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'test.pkl',
        img_prefix=data_root + 'traindataset/train_image/',             #####################
        img_scale=(600, 960),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)         ############################
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[ 20])                                                #################################################
checkpoint_config = None # dict(interval=1) # epoch interval
# yapf:disable
log_config = dict(
    interval=2500,                                                 ##############################
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# kittiEvalAPHook config
kitti_ap_hook_cfg = dict(
    eval_cpg_path = 'tools/kitti_evaluate/evaluate_object',
    dataset_root = dataset_root,
    label_vocab = label_vocab,
    interval = 1,                    ############
    skip_epoch = False,                  #跳过前skip_epoch_num个epoch的eval（第一个epoch不跳过），设为False则每个epoch都会在val集上做测试
    skip_epoch_num = 18,
    skip_epoch_interval = 10            #跳过前skip_epoch_num个epoch的eval中每skip_epoch_interval个epoch eval一个epoch
)
# yapf:enable
import time
# runtime settings
total_epochs = 1000 # 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
tid = time.strftime("%Y%m%d_%H%M%S", time.localtime())
work_dir = './work_dirs/test_out_'+tid                     ######################################
load_from = './work_dirs/with_8_coo_cont_5_3_4_20190310_124543/epoch_4_Hanzi_accuracy0.974307_avr_max_iou0.921272_avr_ratio0.961986.pth'
resume_from = None
workflow = [('train', 1)]
