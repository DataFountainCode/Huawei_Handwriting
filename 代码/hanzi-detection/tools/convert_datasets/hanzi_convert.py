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
    parser.add_argument('--devkit_path', default='/home/chenriquan/Datasets/hanzishibie/', help='hanzishibie Detection devkit path')
    parser.add_argument('--train_data_num', default=50004, help='training_data_num')
    args = parser.parse_args()
    return args

TYPE2LABEL = dict(
    Background=0,
    Hanzi=1,

)


def main():
    args = parse_args()

    devkit_path = args.devkit_path

    # preprocessing training annotation file
    ann_dir = os.path.join(args.devkit_path, 'traindataset','train_label')
    data_dir = os.path.join(args.devkit_path,'traindataset', 'train_image')
    train_ann = []
    print("Preprocessing training annotation...")
    for i in range(1, 60001):
        ann_filename = str(i).zfill(6)+'.txt'
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
    data_dir = os.path.join(args.devkit_path, 'test_dataset')
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

    splited_train = [train_ann[x] for x in tr_idx[:args.train_data_num]]
    splited_val = [train_ann[x] for x in tr_idx[args.train_data_num:]]    
    
    out_dir = os.path.join(args.devkit_path, 'train_&_val_pkl')
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(test_ann, os.path.join(out_dir, 'test.pkl'))

    mmcv.dump(splited_train, os.path.join(out_dir, 'train.pkl'))
    mmcv.dump(splited_val, os.path.join(out_dir, 'val.pkl'))

   
    """
    splited_train = [train_ann[x] for x in tr_idx[val_div_i:]]
    splited_val = [train_ann[x] for x in tr_idx[:val_div_i]]
   
    out_dir = os.path.join(args.devkit_path, 'split_'+args.ratio+"1_2_4_1")
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(test_ann, os.path.join(out_dir, 'test.pkl'))
    mmcv.dump(mini_train, os.path.join(out_dir, 'mini_train.pkl'))
    mmcv.dump(mini_val, os.path.join(out_dir, 'mini_val.pkl'))
    mmcv.dump(splited_train, os.path.join(out_dir, 'train.pkl'))
    mmcv.dump(splited_val, os.path.join(out_dir, 'val.pkl'))
    mmcv.dump(val_data, os.path.join(out_dir, 'val_staic.pkl'))
    

    np.random.shuffle(tr_idx)
    splited_train = [train_ann[x] for x in tr_idx[val_div_i:]]
    splited_val = [train_ann[x] for x in tr_idx[:val_div_i]]
    
    out_dir = os.path.join(args.devkit_path, 'split_'+args.ratio+"1_2_4_2")
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(test_ann, os.path.join(out_dir, 'test.pkl'))
    mmcv.dump(mini_train, os.path.join(out_dir, 'mini_train.pkl'))
    mmcv.dump(mini_val, os.path.join(out_dir, 'mini_val.pkl'))
    mmcv.dump(splited_train, os.path.join(out_dir, 'train.pkl'))
    mmcv.dump(splited_val, os.path.join(out_dir, 'val.pkl'))
    mmcv.dump(val_data, os.path.join(out_dir, 'val_staic.pkl'))

    np.random.shuffle(tr_idx)
    splited_train = [train_ann[x] for x in tr_idx[val_div_i:]]
    splited_val = [train_ann[x] for x in tr_idx[:val_div_i]]
    
    out_dir = os.path.join(args.devkit_path, 'split_'+args.ratio+"1_2_4_3")
    mmcv.mkdir_or_exist(out_dir)
    mmcv.dump(test_ann, os.path.join(out_dir, 'test.pkl'))
    mmcv.dump(mini_train, os.path.join(out_dir, 'mini_train.pkl'))
    mmcv.dump(mini_val, os.path.join(out_dir, 'mini_val.pkl'))
    mmcv.dump(splited_train, os.path.join(out_dir, 'train.pkl'))
    mmcv.dump(splited_val, os.path.join(out_dir, 'val.pkl'))
    mmcv.dump(val_data, os.path.join(out_dir, 'val_staic.pkl'))
    """

    print('[Total img]')
    print('train: ', len(splited_train))
    print('val: ', len(splited_val))
    print('test: ', len(test_ann))


    print('Done!')


if __name__ == '__main__':
    main()
