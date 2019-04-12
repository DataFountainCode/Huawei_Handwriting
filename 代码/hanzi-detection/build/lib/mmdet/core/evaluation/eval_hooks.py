import os
import os.path as osp
import shutil
import time

import mmcv
import numpy as np
import torch
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

from .coco_utils import results2json, fast_eval_recall
from mmdet import datasets


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.lock_dir = None

    def _barrier(self, rank, world_size):
        """Due to some issues with `torch.distributed.barrier()`, we have to
        implement this ugly barrier function.
        """
        if rank == 0:
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                while not (osp.exists(tmp)):
                    time.sleep(1)
            for i in range(1, world_size):
                tmp = osp.join(self.lock_dir, '{}.pkl'.format(i))
                os.remove(tmp)
        else:
            tmp = osp.join(self.lock_dir, '{}.pkl'.format(rank))
            mmcv.dump([], tmp)
            while osp.exists(tmp):
                time.sleep(1)

    def before_run(self, runner):
        self.lock_dir = osp.join(runner.work_dir, '.lock_map_hook')
        if runner.rank == 0:
            if osp.exists(self.lock_dir):
                shutil.rmtree(self.lock_dir)
            mmcv.mkdir_or_exist(self.lock_dir)

    def after_run(self, runner):
        if runner.rank == 0:
            shutil.rmtree(self.lock_dir)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            for _ in range(batch_size):
                prog_bar.update()

        if runner.rank == 0:
            print('\n')
            self._barrier(runner.rank, runner.world_size)
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            self._barrier(runner.rank, runner.world_size)
        self._barrier(runner.rank, runner.world_size)

    def evaluate(self):
        raise NotImplementedError


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            field = '{}_mAP'.format(res_type)
            runner.log_buffer.output[field] = cocoEval.stats[0]
        runner.log_buffer.ready = True
        os.remove(tmp_file)

class KittiEvalAPHook(Hook):
    def __init__(self, dataset_cfg, eval_cpg_path, dataset_root, label_vocab, interval):
        self.dataset = obj_from_dict(dataset_cfg, datasets,
                                        {'test_mode': True})
        self.eval_cpg_path = eval_cpg_path
        self.dataset_root = dataset_root
        self.label_vocab = label_vocab
        self.interval = interval
        self.writer = None
        self.tf_iter = 0

    def after_train_iter(self, runner):
        """Compute AP here with C++ program in path
        """

        if (self.every_n_inner_iters(runner, self.interval)):
            init_start = time.time()
            eval_time = 0.0

            # prepared cpp program, gt label soft link
            script_cpg_path = os.path.join(runner.work_dir, os.path.basename(self.eval_cpg_path))
            if not os.path.exists(script_cpg_path):
                os.system("cp %s %s"%(self.eval_cpg_path, script_cpg_path))

            label_path = os.path.join(runner.work_dir, 'data_object_label_2')
            if not os.path.exists(label_path):
                os.system("ln -s %s %s"%(os.path.join(self.dataset_root, 'data_object_label_2'), label_path))

            results_path = os.path.join(runner.work_dir, 'results')
            if not os.path.exists(results_path):
                os.mkdir(results_path)
            eval_tid = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            results_path = os.path.join(results_path, eval_tid)
            if not os.path.exists(results_path):
                os.mkdir(results_path)
            results_data_path = os.path.join(results_path, 'data')
            if not os.path.exists(results_data_path):
                os.mkdir(results_data_path)

            # use model eval result first
            runner.model.eval()
            prog_bar = mmcv.ProgressBar(len(self.dataset))
            for idx in range(len(self.dataset)):
                eval_start = time.time()
                data = self.dataset[idx]
                data_gpu = scatter(collate([data], samples_per_gpu=1),[torch.cuda.current_device()])[0]
                # compute output
                with torch.no_grad():
                    result = runner.model(
                        return_loss=False, rescale=True, **data_gpu)
                eval_time += time.time() - eval_start
                # save prediction
                pred_fn = self.dataset.img_infos[idx]['filename'].split('.')[0] + '.txt'
                pred_path = os.path.join(results_data_path, pred_fn)
                with open(pred_path, 'w') as f:
                    for l, re in enumerate(result):
                        label_name = self.label_vocab[l+1]
                        for pred_bb in re:
                            formated_det = "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n"%(
                                label_name, -1, -1, -10, \
                                pred_bb[0], pred_bb[1], pred_bb[2], pred_bb[3], \
                                -1, -1, -1, -1000, -1000, -1000, \
                                -10, pred_bb[4]
                            )
                            f.write(formated_det)

                prog_bar.update()
            
            # run cpp script to evalute precise-recall plot
            print("")
            os.system('cd %s && ./evaluate_object %s'%(runner.work_dir, eval_tid))
            
            # compute AP from 41 points precision stat file
            iou_thr = [0.7, 0.5, 0.5]
            levels = ['E', 'M', 'H']
            level_AP = {}
            for i in range(1, len(self.label_vocab)):
                label_name = self.label_vocab[i]
                stats_file_path = os.path.join(results_path, 'stats_%s_detection.txt'%label_name)

                level_AP[label_name] = dict(E=0.0, M=0.0, H=0.0)  # Hard Moderate Easy
                if os.path.exists(stats_file_path):
                    with open(stats_file_path, 'r') as f:
                        for li, line in enumerate(f.readlines()):
                            level = levels[li]
                            det_41p_prec = [float(x) for x in line.split(' ')[:-1]]
                            level_AP[label_name][level] = sum(det_41p_prec)
                            level_AP[label_name][level] /= len(det_41p_prec)
                    print("%-18s: [Easy: %2.4f] [Moderate: %2.4f] [Hard: %2.4f]"%( \
                        label_name[0].upper()+label_name[1:]+" @AP%.1f"%iou_thr[i-1], \
                        level_AP[label_name]['E'], level_AP[label_name]['M'], level_AP[label_name]['H']))

            # log to tb
            if self.writer is None:
                self.writer = SummaryWriter(os.path.join(runner.work_dir, 'tf_logs'))
            self.tf_iter += 1
            for i, tag in enumerate(level_AP):
                lv_ap = level_AP[tag]
                tb_tag = 'AP_eval/'+tag
                self.writer.add_scalars(tb_tag, lv_ap, self.tf_iter)

            # save checkpoint
            runner.save_checkpoint(out_dir=runner.work_dir, \
                filename_tmpl='epoch_{}_car_E%.3f_M%.3f_H%.3f.pth'%(level_AP['car']['E'], level_AP['car']['M'], level_AP['car']['H']),\
                save_optimizer=True)

            # time cost
            print("eval_time %fs(total), %fs(per img)"%(time.time()-init_start, eval_time/len(self.dataset)))

