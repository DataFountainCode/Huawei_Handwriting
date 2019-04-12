import argparse
import os.path as osp
import os
import shutil

import mmcv
import numpy as np
import cv2
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visual result detection demo')
    parser.add_argument('--pred_anno', help='KITTI Detection devkit dir')
    parser.add_argument('--gt_anno', default='/home/chenriquan/Datasets/KITTIdevkit/KITTI/data_object_label_2/training/label_2', help='KITTI Detection devkit dir')
    parser.add_argument('-o', '--out_dir', default='demo', help='output path')
    parser.add_argument('-i', '--img_dir', default='/home/chenriquan/Datasets/KITTIdevkit/KITTI/data_object_image_2/training/image_2', help='output path')
    parser.add_argument('--test', action='store_true', help='whether to evaluate gt')
    args = parser.parse_args()
    return args


vis_cfg = dict(
    bbox_color='blue',
    text_color='yellow',
    font_scale=0.3,
    show=False
)
N_TESTIMAGES = 7518
def main():
    args = parse_args()
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.mkdir(args.out_dir)

    pred_num = 0
    for i in tqdm(range(N_TESTIMAGES)):
        basename = "%06d"%i
        if os.path.exists(os.path.join(args.pred_anno, basename+'.txt')):
            img_gt = cv2.imread(os.path.join(args.img_dir, basename+'.png'))
            img_pred = img_gt.copy()
            with open(os.path.join(args.pred_anno, basename+'.txt'), 'r') as f:
                bboxes = []
                labels = []
                class_names = []
                for l in f.readlines():
                    str_split = l.split(' ')
                    label = str_split[0]
                    x1, y1, x2, y2 = float(str_split[4]), float(str_split[5]), float(str_split[6]), float(str_split[7])
                    bboxes.append([x1, y1, x2, y2])
                    conf = float(str_split[-1])
                    text = '%s(%.3f)'%(label, conf)

                    class_names.append(text)
                    labels.append(len(class_names)-1)
                if len(bboxes) != 0:
                    mmcv.visualization.imshow_det_bboxes(img_pred, np.array(bboxes), np.array(labels, dtype=np.int32), class_names, **vis_cfg)


            if not args.test:
                with open(os.path.join(args.gt_anno, basename+'.txt'), 'r') as f:
                    bboxes = []
                    labels = []
                    class_names = []
                    for l in f.readlines():
                        str_split = l.split(' ')
                        label = str_split[0].lower()
                        class_names.append(label)

                        x1, y1, x2, y2 = float(str_split[4]), float(str_split[5]), float(str_split[6]), float(str_split[7])
                        bboxes.append([x1, y1, x2, y2])
                        labels.append(len(class_names)-1)

                    if len(bboxes) != 0:
                        mmcv.visualization.imshow_det_bboxes(img_gt, np.array(bboxes), np.array(labels, dtype=np.int32), class_names, **vis_cfg)        

                cv2.imwrite(os.path.join(args.out_dir, basename+'.png'), np.vstack([img_gt, img_pred]))
            else:
                cv2.imwrite(os.path.join(args.out_dir, basename+'.png'), img_pred)
            pred_num += 1
    
    print("Total demo: %d"%pred_num)

if __name__ == "__main__":
    main()