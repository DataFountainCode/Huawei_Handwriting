from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
from dataset import *
import models.crnn as crnn
import re
import test_params
import logging
import os
import time
import sys
from tensorboardX import SummaryWriter
from tqdm import tqdm
import mmcv

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--GPU_ID', type=int, default=None, help='GPU_ID')

opt = parser.parse_args()
val_epoch = 0

def get_log_dir():
    run_id = test_params.name+f'_lr_{test_params.lr:.5f}_batchSize_{test_params.batchSize:d}_time_%s_'%time.strftime('%m%d%H%M%S')+'/'
    log_dir = os.path.join(test_params.log_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger
    
# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def val(net, dataset, criterion, epoch, step, max_iter=100000):
    logger.info('Start val')
    # for p in crnn.parameters():
    #     p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=1, num_workers=int(test_params.workers), collate_fn=alignCollate(imgH=test_params.imgH, imgW=test_params.imgW, keep_ratio=test_params.keep_ratio))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    max_iter = len(data_loader)
    record_dir = log_dir + 'epoch_%d_step_%d_data.txt'%(epoch, step)
    r = 1
    f = open(record_dir, "a")
    num_label, num_pred = test_params.total_num, 0

    start = time.time()
    prog_bar = mmcv.ProgressBar(max_iter)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        with torch.no_grad():
            preds = crnn(image)

        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        print(preds)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        if not isinstance(sim_preds, list):
            sim_preds = [sim_preds]
        for pred in sim_preds:
            f.write(str(r).zfill(6)+".jpg "+pred+"\n")
            r += 1
        list_1 = []
        for i in cpu_texts:
            string = i.decode('utf-8', 'strict')
            list_1.append(string)     
        for pred, target in zip(sim_preds, list_1):
            if pred == target:
                n_correct += 1

        num_pred += len(sim_preds)

        prog_bar.update()
    print("")
    f.close()

    
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:test_params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
        logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    logger.info('correct_num: %d'%(n_correct))
    logger.info('Total_num: %d'%(max_iter*test_params.batchSize))
    accuracy = float(n_correct) / num_pred
    recall = float(n_correct) / num_label
    logger.info('Test loss: %f, accuray: %f, recall: %f, F1 score: %f, Cost : %.4fs per img'
                % (loss_avg.val(), accuracy, recall, 2*accuracy*recall/(accuracy+recall+1e-2), (time.time()-start)/max_iter))

    global val_epoch
    writer.add_scalar("val/loss", loss_avg.val(), val_epoch)
    writer.add_scalar("val/acc", accuracy, val_epoch)
    writer.add_scalar("val/recall", recall, val_epoch)
    writer.add_scalar("val/F1", 2*accuracy*recall/(accuracy+recall+1e-2), val_epoch)
    val_epoch += 1


def trainBatch(net, criterion, optimizer, train_iter):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost
    
def training():
    for total_steps in range(test_params.niter):
        train_iter = iter(train_loader)
        i = 0
        logger.info('length of train_data: %d'%(len(train_loader)))

        eval_time = 0.0
        prog_bar = mmcv.ProgressBar(test_params.displayInterval)
        while i < len(train_loader):
            torch.cuda.empty_cache()
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            val(crnn, test_dataset, criterion, total_steps, i)
            return
            start = time.time()
            cost = trainBatch(crnn, criterion, optimizer, train_iter)
            eval_time += time.time()-start

            loss_avg.add(cost)
            i += 1
            prog_bar.update()


            if i % test_params.tbInterval == 0:
                print("\n>>>> Tensorboard Log")
                writer.add_scalar('train/loss', loss_avg.val(), int(i+total_steps*len(train_loader)))  # record to tb

            if i % test_params.displayInterval == 0:
                sys.stdout.write("\r%100s\r"%' ')
                sys.stdout.flush()
                logger.info('[%d/%d][%d/%d] Loss: %f, Cost: %.4fs per batch' %
                      (total_steps, test_params.niter, i, len(train_loader), loss_avg.val(), eval_time/i))

                if eval_time/i < 0.2: test_params.displayInterval = 1000
                elif eval_time/i < 0.5: test_params.displayInterval = 400
                elif eval_time/i < 1.0: test_params.displayInterval = 200
                prog_bar = mmcv.ProgressBar(test_params.displayInterval)  # new interval

                loss_avg.reset()
 
                

        val(crnn, test_dataset, criterion, total_steps, i)
        torch.cuda.empty_cache()
        if (total_steps+1) % test_params.saveInterval == 0:
            string = "model save to {0}crnn_Rec_done_epoch_{1}.pth".format(log_dir, total_steps)
            logger.info(string)
            torch.save(crnn.state_dict(), '{0}crnn_Rec_done_epoch_{1}.pth'.format(log_dir, total_steps))

if __name__ == '__main__':
    
    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    log_dir = get_log_dir()
    logger = get_logger(log_dir, test_params.name, test_params.name+'_info.log')
    logger.info(opt)

    # tensorboardX
    writer = SummaryWriter(os.path.join(log_dir, 'tb_logs'))

    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU_ID)
    # read train set
    train_dataset = dataset.lmdbDataset(root=opt.trainroot, rgb=test_params.rgb, rand_hcrop=True)
    assert train_dataset
    if test_params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, test_params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*160
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=test_params.batchSize,
        shuffle=False, sampler=sampler,
        num_workers=int(test_params.workers),
        collate_fn=dataset.alignCollate(imgH=test_params.imgH, imgW=test_params.imgW, keep_ratio=test_params.keep_ratio))
    
    # read test set
    # images will be resize to 32*160
    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, rgb=test_params.rgb)

    nclass = len(test_params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(test_params.alphabet)
    criterion = CTCLoss()

    # cnn and rnn
    image = torch.FloatTensor(test_params.batchSize, 1, test_params.imgH, test_params.imgH)
    text = torch.IntTensor(test_params.batchSize * 5)
    length = torch.IntTensor(test_params.batchSize)

    crnn = crnn.CRNN(test_params.imgH, nc, nclass, test_params.nh)
    if opt.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)
    if test_params.crnn != '':
        logger.info('loading pretrained model from %s' % test_params.crnn)
        if test_params.without_fully:
            pretrained_dict = torch.load(test_params.crnn)
            model_dict=crnn.state_dict()
            pretrained_dict.pop('rnn.1.embedding.weight')
            pretrained_dict.pop('rnn.1.embedding.bias')    
            crnn.load_state_dict(pretrained_dict, strict=False)    
        else:    
            crnn.load_state_dict(torch.load(test_params.crnn), strict=False)

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if test_params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=test_params.lr,
                               betas=(test_params.beta1, 0.999))
    elif test_params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=test_params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=test_params.lr)
    #optimizer = lr_scheduler.StepLR(optimizer, step_size=25*len(train_loader), gamma=0.1)    

    training()
