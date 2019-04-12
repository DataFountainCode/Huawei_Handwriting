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
import models.crnn as crnn
import re
import params
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
val_epoch = 0

def get_log_dir():
    run_id = params.name+f'_lr_{params.lr:.5f}_batchSize_{params.batchSize:d}_time_%s_'%time.strftime('%m%d%H%M%S')+'/'
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

def val(net, dataset, criterion, epoch, step, max_iter=100000):
    logger.info('Start val')
    # for p in crnn.parameters():
    #     p.requires_grad = False
    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=1, num_workers=int(params.workers), collate_fn=alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    max_iter = len(data_loader)
    record_dir = log_dir + 'epoch_%d_step_%d_data.txt'%(epoch, step)
    r = 1
    f = open(record_dir, "a")
    num_label, num_pred = params.total_num, 0

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

    
    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
        logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    logger.info('correct_num: %d'%(n_correct))
    logger.info('Total_num: %d'%(max_iter*params.batchSize))
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

    preds = net(image)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    cost = criterion(preds, text, preds_size, length) / batch_size
    net.zero_grad()
    cost.backward()
    optimizer.step()
    '''
    run_ok = False
    try:
        data = train_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = net(image)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        net.zero_grad()
        cost.backward()
        optimizer.step()
        run_ok = True
    except RuntimeError as e:
        raise RuntimeError("RuntimeError: data size (%d %d %d %d)"%cpu_images.size())
    finally:
        if not run_ok:
            raise RuntimeError("FinallyError: data size (%d %d %d %d)"%cpu_images.size())
    '''
    return cost
    
def training():
    for total_steps in range(params.niter):
        train_iter = iter(train_loader)
        i = 0
        logger.info('length of train_data: %d'%(len(train_loader)))

        eval_time = 0.0
        prog_bar = mmcv.ProgressBar(params.displayInterval)
        while i < len(train_loader):
            i += 1
            runtime_error = False
            crnn.train()
            loss_avg.reset()
            start = time.time()
            cost = trainBatch(crnn, criterion, optimizer, train_iter)
            eval_time += time.time()-start
            loss_avg.add(cost.cpu())
            prog_bar.update()
            '''
            try:
                i += 1

                #crnn.cuda()
                crnn.train()
                loss_avg.reset()
                start = time.time()
                cost = trainBatch(crnn, criterion, optimizer, train_iter)
                eval_time += time.time()-start
                loss_avg.add(cost.cpu())
                prog_bar.update()
 
                runtime_error = False
            except RuntimeError as e:
                logger.error(e)
                runtime_error = True
            except ConnectionRefusedError as e:
                logger.error(e)
                runtime_error = True
            finally:
                if runtime_error:
                    logger.error("Warning: Some error happen")
                    gc.collect()
                    torch.cuda.empty_cache()
            '''

            if i % params.tbInterval == 0 and not runtime_error:
                print("\n>>>> Tensorboard Log")
                writer.add_scalar('train/loss', loss_avg.val(), int(i+total_steps*len(train_loader)))  
                # record to tb

            if i % params.displayInterval == 0 and not runtime_error:
                sys.stdout.write("\r%100s\r"%' ')
                sys.stdout.flush()
                logger.info('[%d/%d][%d/%d] Loss: %f, Cost: %.4fs per batch' %
                      (total_steps, params.niter, i, len(train_loader), loss_avg.val(), eval_time/i))
                loss_avg.reset()
                if eval_time/i < 0.2: params.displayInterval = 1000
                elif eval_time/i < 0.5: params.displayInterval = 400
                elif eval_time/i < 1.0: params.displayInterval = 200
                prog_bar = mmcv.ProgressBar(params.displayInterval)  # new interval
              
                


            # if i % params.valInterval == 0:
            #     val(crnn, test_dataset, criterion, total_steps, i)
            #     torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        val(crnn, test_dataset, criterion, total_steps, i)

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

    # tensorboardX
    writer = SummaryWriter(os.path.join(log_dir, 'tb_logs'))

    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    # read train set
    tr_t = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.1, hue=0.1),
    ])
    train_dataset = dataset.lmdbDataset(root=opt.trainroot, rgb=params.rgb, transform=tr_t, rand_hcrop=True)
    assert train_dataset
    if params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*160
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params.batchSize,
        shuffle=False, sampler=sampler,
        num_workers=int(params.workers),
        collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    train_iter = iter(train_loader)
    for i in range(5000):
        print(train_iter.next()[0].shape)
    # read test set
    # images will be resize to 32*160
    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, rgb=params.rgb)

    nclass = len(params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(params.alphabet)
    criterion = CTCLoss(size_average=False, length_average=False)

    # cnn and rnn
    image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
    text = torch.IntTensor(params.batchSize * 5)
    length = torch.IntTensor(params.batchSize)

    crnn = crnn.CRNN(params.imgH, nc, nclass, params.nh)
    crnn = torch.nn.DataParallel(crnn)
    if opt.cuda:
        crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    crnn.apply(weights_init)
    if params.crnn != '':
        logger.info('loading pretrained model from %s' % params.crnn)
        crnn.load_state_dict(torch.load(params.crnn), strict=False)

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
    #optimizer = lr_scheduler.StepLR(optimizer, step_size=25*len(train_loader), gamma=0.1)    

    training()
