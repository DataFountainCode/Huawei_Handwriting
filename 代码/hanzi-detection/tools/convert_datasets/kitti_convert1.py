import argparse
import os.path as osp
import os

import mmcv
import numpy as np
import cv2
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert KITTI Detection annotations to mmdetection format')
    parser.add_argument('--devkit_path', default='/home/chenriquan/Datasets/KITTIdevkit/KITTI/', help='KITTI Detection devkit path')
    parser.add_argument('-r', '--ratio', help='valid : train ratio e.g 1:5')
    args = parser.parse_args()
    return args

TYPE2LABEL = dict(
    Background=0,
    Car=1,
    # Cyclist=2,
    # Pedestrian=3,
    Van=1,
    # Person_sitting=3
)


def main():
    args = parse_args()
    if args.ratio is None:
        print("args.ratio is needed!!")
        return
    print("val:train = %d:%d"%(int(args.ratio.split(':')[0]), int(args.ratio.split(':')[1])))
    val_train_ratio = int(args.ratio.split(':')[0]) / int(args.ratio.split(':')[1])
    devkit_path = args.devkit_path

    # preprocessing training annotation file
    ann_dir = os.path.join(args.devkit_path, 'data_object_label_2', 'training', 'label_2')
    data_dir = os.path.join(args.devkit_path, 'data_object_image_2', 'training', 'image_2')
    train_ann = []
    print("Preprocessing training annotation...")
    for ann_filename in tqdm(os.listdir(ann_dir)):
        im_filename = ann_filename.split('.')[0] + '.png'
        im = cv2.imread(os.path.join(data_dir, im_filename))
        bboxes = []
        bboxes_ignore = []
        labels = []
        with open(os.path.join(ann_dir, ann_filename), 'r') as f:
            for line in f.readlines():
                str_ann = line.split(' ')
                label_type = str_ann[0]
                label_bbox = np.array([float(x) for x in str_ann[4:8]])
                if label_type in TYPE2LABEL:
                    bboxes.append(label_bbox)
                    labels.append(TYPE2LABEL[label_type])
                if label_type == 'DontCare':
                    bboxes_ignore.append(label_bbox)
        train_ann.append(dict(
            filename=im_filename,
            width=im.shape[1],
            height=im.shape[0],
            ann=dict(
                bboxes=np.array(bboxes, dtype=np.float32),
                labels=np.array(labels, dtype=np.int64),
                bboxes_ignore=np.array(bboxes_ignore),
                labels_ignore=None
            )
        ))

    # preprocessing testing annotation file
    data_dir = os.path.join(args.devkit_path, 'data_object_image_2', 'testing', 'image_2')
    test_ann = []
    print("Preprocessing testing annotation...")
    for im_filename in tqdm(os.listdir(data_dir)):
        im = cv2.imread(os.path.join(data_dir, im_filename))
        test_ann.append(dict(
            filename=im_filename,
            width=im.shape[1],
            height=im.shape[0]
        ))

                
    # split train, validate and test annotation
    tr_idx = np.arange(len(train_ann))
    mini_tr_len = 1000  # for debug
    mini_val_len = 200
    np.random.shuffle(tr_idx)
    
    mini_train = [train_ann[x] for x in tr_idx[:mini_tr_len]]
    mini_val = [train_ann[x] for x in tr_idx[mini_tr_len:mini_tr_len+mini_val_len]]

    np.random.shuffle(tr_idx)
    val_ratio = val_train_ratio/(val_train_ratio+1)
    val_div_i = int(tr_idx.shape[0]*val_ratio*4/7)
    end=int(tr_idx.shape[0]*4/7)
    static_idx=tr_idx[end:]
    tr_idx=tr_idx[:end]

    val_data=[train_ann[x] for x in static_idx]

    

    splited_train = [train_ann[x] for x in tr_idx[val_div_i:]]
    splited_val = [train_ann[x] for x in tr_idx[:val_div_i]]
   
    out_dir = os.path.join(args.devkit_path, 'split_'+args.ratio+"1:3:3")
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(test_ann, os.path.join(out_dir, 'test.pkl'))
    mmcv.dump(mini_train, os.path.join(out_dir, 'mini_train.pkl'))
    mmcv.dump(mini_val, os.path.join(out_dir, 'mini_val.pkl'))
    mmcv.dump(splited_train, os.path.join(out_dir, 'train.pkl'))
    mmcv.dump(splited_val, os.path.join(out_dir, 'val.pkl'))
    mmcv.dump(val_data, os.path.join(out_dir, 'val_staic.pkl'))
    



    print('[Total img]')
    print('train: ', len(splited_train))
    print('val: ', len(splited_val))
    print('test: ', len(test_ann))
    print('-- mini --')
    print('mini train: ', len(mini_train))
    print('mini val: ', len(mini_val))

    print('Done!')


if __name__ == '__main__':
    main()
