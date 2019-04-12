from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn
import re
import sig_model_ensemble_params as params
import logging
import os
import time
import sys
from tensorboardX import SummaryWriter
from tqdm import tqdm
import mmcv
import gc
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--GPU_ID', type=str, default=None, help='GPU_ID')
opt = parser.parse_args()
print(opt)
def get_log_dir():
    run_id = params.name+f'_lr_{params.lr:.7f}_batchSize_{params.batchSize:d}_time_%s_'%time.strftime('%m%d%H%M%S')+'/'
    log_dir = os.path.join(params.log_dir, run_id)
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


def val(net, _dataset1, _dataset2, _dataset3, epoch, step, criterion, max_iter=100):
    logger.info('Start val')
    # for p in crnn.parameters():
    #     p.requires_grad = False
    net.eval()
    data_loader1 = torch.utils.data.DataLoader(
        _dataset1, shuffle=False, batch_size=params.batchSize, num_workers=int(params.workers), collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    data_loader2 = torch.utils.data.DataLoader(
        _dataset2, shuffle=False, batch_size=params.batchSize, num_workers=int(params.workers), collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    data_loader3 = torch.utils.data.DataLoader(
        _dataset3, shuffle=False, batch_size=params.batchSize, num_workers=int(params.workers), collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    val_iter = iter(data_loader1)
    val_iter2 = iter(data_loader2)
    val_iter3 = iter(data_loader3)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    max_iter = len(data_loader1)
    record_dir = log_dir + 'epoch_%d_step_%d_data.txt'%(epoch, step)
    r = 1
    f = open(record_dir, "a")
    num_label, num_pred = params.total_num, 0

    start = time.time()
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)
        data2 = val_iter2.next()
        cpu_images2, _ = data2
        utils.loadData(image2, cpu_images2)
        data3 = val_iter3.next()
        cpu_images3, _ = data3
        utils.loadData(image3, cpu_images3)
        with torch.no_grad():
            preds = torch.mean(torch.cat([torch.unsqueeze(crnn(image), 0), torch.unsqueeze(crnn(image2), 0), torch.unsqueeze(crnn(image3), 0)], 0), 0)
        print('preds: ', preds.shape)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
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

    print("")
    f.close()

    
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
        logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    logger.info('correct_num: %d'%(n_correct))
    logger.info('Total_num: %d'%(max_iter*params.batchSize))
    accuracy = float(n_correct) / num_pred
    recall = float(n_correct) / num_label
    logger.info('Test loss: %f, accuray: %f, recall: %f, F1 score: %f, Cost : %.4fs per img'
                % (loss_avg.val(), accuracy, recall, 2*accuracy*recall/(accuracy+recall+1e-2), (time.time()-start)/max_iter))



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
    for total_steps in range(params.niter):
        train_iter = iter(train_loader)
        i = 0
        logger.info('length of train_data: %d'%(len(train_loader)))
        while i < len(train_loader):
         
            for p in crnn.parameters():
                p.requires_grad = True
                 
            crnn.train()
            val(crnn, test_dataset1, test_dataset2, test_dataset3, total_steps, i, criterion)
            return
            cost = trainBatch(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            if i % params.displayInterval == 0:
                logger.info('[%d/%d][%d/%d] Loss: %f' %
                      (total_steps, params.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()
        val(crnn, test_dataset, total_steps, i, criterion)        
        if (total_steps+1) % params.saveInterval == 0:
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
    logger = get_logger(log_dir, params.name, params.name+'_info.log')
    logger.info(opt)
    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')

    # read train set
    train_dataset = dataset.lmdbDataset(root=opt.trainroot, rand_hcrop=params.with_crop)
    assert train_dataset
    if params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    # images will be resize to 32*160
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batchSize,
        shuffle=False, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    # read test set
    # images will be resize to 32*160
    test_dataset1 = dataset.lmdbDataset3(
        root=opt.valroot, mode='up')
    test_dataset2 = dataset.lmdbDataset3(
        root=opt.valroot, mode='mid')
    test_dataset3 = dataset.lmdbDataset3(
        root=opt.valroot, mode='down')    
    nclass = len(params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(params.alphabet)
    criterion = CTCLoss()

    # cnn and rnn
    image = torch.FloatTensor(params.batchSize, 1, params.imgH, params.imgH)
    text = torch.IntTensor(params.batchSize * 5)
    length = torch.IntTensor(params.batchSize)

    crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
    if opt.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)
    if params.crnn != '':
        logger.info('loading pretrained model from %s' % params.crnn)
        crnn.load_state_dict(torch.load(params.crnn))

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if params.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                               betas=(params.beta1, 0.999))
    elif params.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=params.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

    training()
