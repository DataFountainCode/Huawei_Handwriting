from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.checkpoint import checkpoint as torch_cp
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
from dataset import *
from models.resnet_crnn import ResNetCRNN
import re
import rc_params
import logging
import os
import time
import sys
from tensorboardX import SummaryWriter
from tqdm import tqdm
import mmcv
import torchvision.transforms as transforms
import gc
import math

def main():
    resnet_crnn = ResNetCRNN(rc_params.imgH, 1, len(rc_params.alphabet) + 1, rc_params.nh,
                             resnet_type=rc_params.resnet_type,
                             feat_size=rc_params.feat_size)
    resnet_crnn = torch.nn.DataParallel(resnet_crnn)
    state_dict = torch.load('./work_dirs/resnet18_rcnn_sgd_imgh128_rgb_512x1x16_lr_0.00100_batchSize_8_time_0319110013_/crnn_Rec_done_epoch_7.pth')
    resnet_crnn.load_state_dict(state_dict)
    test_dataset = dataset.lmdbDataset(root='to_lmdb/test_index', rgb=True)
    converter = utils.strLabelConverter(rc_params.alphabet)

    resnet_crnn.eval()
    resnet_crnn.cuda()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=1, num_workers=int(rc_params.workers),
        collate_fn=alignCollate(
            imgH=rc_params.imgH,
            imgW=rc_params.imgW,
            keep_ratio=rc_params.keep_ratio,
            rgb=True)
    )
    val_iter = iter(data_loader)
    max_iter = len(data_loader)
    record_dir = 'test_out/test_out.txt'
    r = 1
    f = open(record_dir, "a")

    image = torch.FloatTensor(rc_params.batchSize, 3, rc_params.imgH, rc_params.imgH)
    prog_bar = mmcv.ProgressBar(max_iter)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        # image = cpu_images.cuda()

        with torch.no_grad():
            preds = resnet_crnn(image)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        if not isinstance(sim_preds, list):
            sim_preds = [sim_preds]
        for pred in sim_preds:
            f.write(str(r).zfill(6) + ".jpg " + pred + "\n")
            r += 1

        prog_bar.update()
    print("")
    f.close()

if __name__ == '__main__':
    main()
