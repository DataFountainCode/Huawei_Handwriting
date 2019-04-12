import argparse

import os
import sys
import shutil
import time
import subprocess
sys.path.append("/home/chenriquan/Projects/kitti-detection/")

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors


def single_test(model, data_loader, gtdir, outdir, label_vocab, eval_cpg_path):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    init_start = time.time()
    eval_time = 0.0

    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for idx in range(len(data_loader.dataset)):
        eval_start = time.time()
        data = data_loader.dataset[idx]
        data_gpu = scatter(collate([data], samples_per_gpu=1),[torch.cuda.current_device()])[0]
        # compute output
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data_gpu)
        
        # save prediction
        pred_fn = data_loader.dataset.img_infos[idx]['filename'].split('.')[0] + '.txt'
        pred_path = os.path.join(outdir, pred_fn)
        with open(pred_path, 'w') as f:
            for l, re in enumerate(result):
                label_name = label_vocab[l+1]
                for pred_bb in re:
                    formated_det = "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n"%(
                        label_name, -1, -1, -10, \
                        pred_bb[0], pred_bb[1], pred_bb[2], pred_bb[3], \
                        -1, -1, -1, -1000, -1000, -1000, \
                        -10, pred_bb[4]
                    )
                    f.write(formated_det)
        prog_bar.update()
    # time cost
    print("\neval_time %fs(total), %fs(per img)"%(time.time()-init_start, eval_time/len(data_loader.dataset)))

    # evaluate
    if len(gtdir) != 0:
        # run cpp script to evalute precise-recall plot
        result_path = os.path.join(outdir, 'result')
        if not os.path.exists(result_path): os.mkdir(result_path)
        if not os.path.exists("./" + os.path.basename(eval_cpg_path)):
            os.system("cp %s ./"%(eval_cpg_path))
        with subprocess.Popen('./%s %s %s %s'%(os.path.basename(eval_cpg_path), outdir, gtdir, result_path), \
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as out_pipe:
            # output format:
            #   Total eval: xx detections files
            #   car (0.7AP|E/M/H) xx xx xx
            #   pedestrian (0.5AP|E/M/H) xx xx xx
            #   cyclist (0.5AP|E/M/H) xx xx xx
            output = out_pipe.stdout.read().decode('utf-8').strip()
            print(output)
        
    return results



def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--outdir', default='test_result', help='output result directory')
    parser.add_argument('--gtdir', default='/home/chenriquan/Datasets/KITTIdevkit/KITTI/data_object_label_2/training/label_2/', help='ground truth for evaluation')
    parser.add_argument('--nms_iou', type=float, default=0)
    parser.add_argument('--max_per_img', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # change nms_iou
    if args.nms_iou != 0:
        cfg.test_cfg['rcnn']['nms']['iou_thr'] = args.nms_iou
    if args.max_per_img != 0:
        cfg.test_cfg['rcnn']['max_per_img'] = args.max_per_img

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, args.gtdir, args.outdir, cfg.label_vocab, cfg.kitti_ap_hook_cfg['eval_cpg_path'])
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)


if __name__ == '__main__':
    main()
