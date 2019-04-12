import argparse
import os, subprocess
import sys
sys.path.append("/home/chenriquan/Projects/kitti-detection/")
import shutil
from mmdet.ops.nms import nms_wrapper
from tqdm import tqdm
import numpy as np

def inner_filter(annos_group, f_thr):
    f_annos_group = [None for _ in range(len(annos_group))]

    iou_cache = dict()
    for i in range(len(f_annos_group)):
        inner_count = None
        for j in range(len(f_annos_group)):
            if i != j:
                cache_key = tuple(sorted([i, j]))
                if cache_key in iou_cache:
                    iou = iou_cache[cache_key]
                    if (i, j) != cache_key:
                        iou = iou.copy().transpose()
                else:
                    iou = bbox_overlaps(annos_group[i], annos_group[j])
                    iou_cache[cache_key] = iou if cache_key == (i, j) else iou.transpose()
                
                # count inner with other bbox under the threshold
                icount = np.sum(iou>f_thr, 1)
                if inner_count is None:
                    inner_count = icount
                else:
                    inner_count += icount
        
        ind = np.where(inner_count>0)[0]
        if len(annos_group[i]) != 0:
            f_annos_group[i] = annos_group[i][ind, :]
        else:
            f_annos_group[i] = annos_group[i][:]

    return f_annos_group

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.

    Refs: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L262
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

def parse_args():
    parser = argparse.ArgumentParser(
        description='Ensemble Detection Result')
    parser.add_argument('--pred_anno_dirs', help='Predict result annotation directory list')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inner_filter', action='store_true')
    parser.add_argument('--new_outdir', action='store_true')
    parser.add_argument('-o', '--out_dir', default='ensembled_pred', help='output directory')
    parser.add_argument('-g', '--gt_dir', default='/home/chenriquan/Datasets/KITTIdevkit/KITTI/data_object_label_2/training/label_2/', help='ground truth directory')
    parser.add_argument('-e', '--eval_script', default='/home/chenriquan/Projects/kitti-detection/tools/kitti_evaluate/evaluate_object', help='evaluate script directory')
    args = parser.parse_args()

    return args

N_TESTIMAGES = 7518

def write_kitti_format(cls_dets, f):
    """Write cls_dets to open f with kitti format
    """
    for l, pred in enumerate(cls_dets):
        formated_det = "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n"%(
            "car", -1, -1, -10, \
            pred[0], pred[1], pred[2], pred[3], \
            -1, -1, -1, -1000, -1000, -1000, \
            -10, pred[4]
        )
        f.write(formated_det)

def evaluate(gtdir, detdir, eval_cpg_path, fp=None):
    # evaluate

    # run cpp script to evalute precise-recall plot
    result_path = os.path.join(detdir, 'result')
    if not os.path.exists(result_path): os.mkdir(result_path)
    if not os.path.exists("./" + os.path.basename(eval_cpg_path)):
        os.system("cp %s ./"%(eval_cpg_path))

    with subprocess.Popen('./%s %s %s %s'%(os.path.basename(eval_cpg_path), detdir, gtdir, result_path), \
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as out_pipe:
        # output format:
        #   Total eval: xx detections files
        #   car (0.7AP|E/M/H) xx xx xx
        #   pedestrian (0.5AP|E/M/H) xx xx xx
        #   cyclist (0.5AP|E/M/H) xx xx xx
        output = out_pipe.stdout.read().decode('utf-8').strip()
        if fp is not None: fp.write(output+"\n\n")
        else: print(output)

    AP = [ float(x) for x in output.split('\n')[1].split(' ')[2:] ]
    AP.append(np.mean(AP))
    return AP


def main():
    args = parse_args()
    gpu_id = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]

    if os.path.exists(args.out_dir):
        if args.new_outdir:
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
    else: os.mkdir(args.out_dir)

    # inner_filter_iou = [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    inner_filter_iou = [0.6]
    if not args.inner_filter:
        inner_filter_iou = [-1]

    # load anno1 and anno2
    annos_f = dict()
    for f_thr in inner_filter_iou:
        annos = dict()
        anno_path_list = args.pred_anno_dirs.split(',')
        for i in tqdm(range(N_TESTIMAGES)):
            basename = "%06d"%i
            img_annos_group = [[] for _ in anno_path_list]
            exists_file_flag = False
            for j, pred_anno_dir in enumerate(anno_path_list):
                anno_path = os.path.join(pred_anno_dir, basename+'.txt')
                if os.path.exists(anno_path):
                    exists_file_flag = True
                    with open(anno_path, 'r') as f:
                        for l in f.readlines():
                            str_split = l.split(' ')
                            x1, y1, x2, y2 = float(str_split[4]), float(str_split[5]), float(str_split[6]), float(str_split[7])
                            box=[x1, y1, x2, y2]
                            conf=[float(str_split[-1])]
                            img_annos_group[j].append(box+conf)
                    
                    img_annos_group[j] = np.array(img_annos_group[j], dtype=np.float32)

            

            if exists_file_flag:
                
                if f_thr != -1:
                    # inner filter
                    img_annos_group = inner_filter(img_annos_group, f_thr)
                
                img_annos_group = [x for x in img_annos_group if x.shape[0] > 0]

                if len(img_annos_group) > 0:
                    img_annos = np.concatenate(img_annos_group, 0)
                else:
                    img_annos = np.array([])
                
                # img_annos = np.array(img_annos, dtype=np.float32)
                if img_annos.shape[0] != 0:
                    img_annos = np.array(img_annos[np.argsort(img_annos[:, 4], ), :])
                annos[i] = img_annos
        
        annos_f[f_thr] = annos
    

    # ensemble with nms or voting
    # nms_iou_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # vot_iou_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # make nms
    nms_iou_list = [0.7]
    vot_iou_list = [0.7]
    
    for f in annos_f:
        ftag = "i%.1f_"%f if f != -1 else ""
        print("inner filter %.1f"%f)
        annos = annos_f[f]
        nms_op = getattr(nms_wrapper, 'nms')
        for key in tqdm(annos):
            basename = "%06d"%key+".txt"
            all_dets = annos[key]
            
            # nms
            for nms_iou in nms_iou_list:
                if all_dets.shape[0] != 0:
                    cls_dets_nms, _ = nms_op(all_dets, iou_thr=nms_iou, device_id=0)
                else:
                    cls_dets_nms = all_dets
                
                out_nms_dir = os.path.join(args.out_dir, '%snms%.1f'%(ftag, nms_iou))
                if not os.path.exists(out_nms_dir): os.mkdir(out_nms_dir)
                new_anno_path = os.path.join(out_nms_dir, basename)
                if not os.path.exists(new_anno_path):
                    with open(new_anno_path, 'w') as f:
                        write_kitti_format(cls_dets_nms, f)
                
                # nms+voting
                for iou in vot_iou_list:   
                    out_vot_nms_dir = os.path.join(args.out_dir, '%svot%.1f_nms%.1f'%(ftag, iou, nms_iou))
                    if not os.path.exists(out_vot_nms_dir): os.mkdir(out_vot_nms_dir)
                    new_anno_path = os.path.join(out_vot_nms_dir, basename)
                    
                    if not os.path.exists(new_anno_path):
                        # If not exist then compute
                        if cls_dets_nms.shape[0] != 0:
                            cls_dets_nms_vot = box_voting(cls_dets_nms, all_dets, iou)
                        else:
                            cls_dets_nms_vot = cls_dets_nms

                        with open(new_anno_path, 'w') as f:
                            write_kitti_format(cls_dets_nms_vot, f)
                
                # voting+nms
                for iou in vot_iou_list:
                    out_nms_vot_dir = os.path.join(args.out_dir, '%snms%.1f_vot%.1f'%(ftag, nms_iou, iou))
                    if not os.path.exists(out_nms_vot_dir): os.mkdir(out_nms_vot_dir)
                    new_anno_path = os.path.join(out_nms_vot_dir, basename)

                    if not os.path.exists(new_anno_path):
                        # If not exist then compute
                        if all_dets.shape[0] != 0:
                            cls_dets_vot = box_voting(all_dets, all_dets, iou)
                            cls_dets_vot_nms, _ = nms_op(cls_dets_vot, iou_thr=nms_iou, device_id=0)
                        else:
                            cls_dets_vot = all_dets
                            cls_dets_vot_nms = all_dets
                        with open(new_anno_path, 'w') as f:
                            write_kitti_format(cls_dets_vot_nms, f)
        

    fp = open(os.path.join(args.out_dir, 'stat_AP.txt'), 'w')
    # evaluate every split
    split_stat = 'split0-2:{}\n'.format(anno_path_list)
    for i, anno in enumerate(anno_path_list):
        if os.path.isdir(anno):
            AP = evaluate(args.gt_dir, anno, args.eval_script, fp)
            split_stat += "split%d (E/M/H/mean): %.6f/%.6f/%.6f/%.6f\n"%(i, AP[0], AP[1], AP[2], AP[3])
    fp.write(split_stat)

    # evaluate for every result
    if args.eval:
        best_AP = dict(best_E=0, best_M=0, best_H=0, best_mAP=0)
        best_AP_e = dict(best_E='', best_M='', best_H='', best_mAP='')
        for endir in tqdm(os.listdir(args.out_dir)):
            fp.write("%s:\n"%endir)
            det_dir = os.path.join(args.out_dir, endir)
            if os.path.isdir(det_dir):
                AP = evaluate(args.gt_dir, det_dir, args.eval_script, fp)
                for i, k in enumerate(best_AP):
                    if AP[i] > best_AP[k]:
                        best_AP[k] = AP[i]
                        best_AP_e[k] = "%s - (%.6f/%.6f/%.6f/%.6f)"%(endir, AP[0], AP[1], AP[2], AP[3])
        print(split_stat)
        for k in best_AP:
            res = "%s (E/M/H/mean): %s\n"%(k, best_AP_e[k])
            print(res, end='')
            fp.write(res)
    
    
    fp.close()


if __name__ == "__main__":
    main()