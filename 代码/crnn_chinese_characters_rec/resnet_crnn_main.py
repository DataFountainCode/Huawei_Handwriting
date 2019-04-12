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

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--GPU_ID', type=str, default=None, help='GPU_ID')

opt = parser.parse_args()
val_epoch = 0


def get_log_dir():
    run_id = rc_params.name + f'_lr_{rc_params.lr:.5f}_batchSize_{rc_params.batchSize:d}_time_%s_' % time.strftime(
        '%m%d%H%M%S') + '/'
    log_dir = os.path.join(rc_params.log_dir, run_id)
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


# custom weights initialization called on resnet_crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(net, dataset, criterion, epoch, step, max_iter=100000):
    logger.info('Start val')
    net.eval()
    net.cuda()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=1, num_workers=int(rc_params.workers),
        collate_fn=alignCollate(
            imgH=rc_params.imgH,
            imgW=rc_params.imgW,
            keep_ratio=rc_params.keep_ratio,
            rgb=True)
    )
    val_iter = iter(data_loader)
    i = 0
    n_correct = 0
    loss_avg = utils.averager()
    max_iter = len(data_loader)
    record_dir = log_dir + 'epoch_%d_step_%d_data.txt' % (epoch, step)
    r = 1
    f = open(record_dir, "a")
    num_label, num_pred = rc_params.total_num, 0

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
            preds = resnet_crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length) / batch_size

        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        if not isinstance(sim_preds, list):
            sim_preds = [sim_preds]
        for pred in sim_preds:
            f.write(str(r).zfill(6) + ".jpg " + pred + "\n")
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

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:rc_params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_1):
        logger.info('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    logger.info('correct_num: %d' % (n_correct))
    logger.info('Total_num: %d' % (max_iter * rc_params.batchSize))
    accuracy = float(n_correct) / num_pred
    recall = float(n_correct) / num_label
    logger.info('Test loss: %f, accuray: %f, recall: %f, F1 score: %f, Cost : %.4fs per img'
                % (loss_avg.val(), accuracy, recall, 2 * accuracy * recall / (accuracy + recall + 1e-2),
                   (time.time() - start) / max_iter))

    global val_epoch
    writer.add_scalar("val/loss", loss_avg.val(), val_epoch)
    writer.add_scalar("val/acc", accuracy, val_epoch)
    writer.add_scalar("val/recall", recall, val_epoch)
    writer.add_scalar("val/F1", 2 * accuracy * recall / (accuracy + recall + 1e-2), val_epoch)
    val_epoch += 1


def trainBatch(net, criterion, optimizer, train_iter):
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
        cost = criterion(preds, text, preds_size, length)
        net.zero_grad()
        cost.backward()
        optimizer.step()
        run_ok = True
    except RuntimeError as e:
        raise RuntimeError("RuntimeError: data size (%d %d %d %d)"%cpu_images.size())
    finally:
        if not run_ok:
            raise RuntimeError("FinallyError: data size (%d %d %d %d)"%cpu_images.size())

    return cost


def training():
    global train_loader
    for total_steps in range(rc_params.niter):
        train_iter = iter(train_loader)
        i = 0
        logger.info('length of train_data: %d' % (len(train_loader)))

        eval_time = 0.0
        inf_num = 0
        prog_bar = mmcv.ProgressBar(rc_params.displayInterval)
        while i < len(train_loader):
            runtime_error = True

            try:
                i += 1
                torch.cuda.empty_cache()
                resnet_crnn.cuda()
                resnet_crnn.train()

                start = time.time()
                cost = trainBatch(resnet_crnn, criterion, optimizer, train_iter)
                eval_time += time.time() - start

                if math.isinf(cost.cpu()):
                    logger.warning("Current loss is INF!!!")
                    inf_num += 1
                    if inf_num > 10:
                        break
                else:
                    loss_avg.add(cost.cpu())
                    inf_num = 0
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

            if i % rc_params.tbInterval == 0 and not runtime_error:
                print("\n>>>> Tensorboard Log")
                writer.add_scalar('train/loss', loss_avg.val(),
                                  int(i + total_steps * len(train_loader)))  # record to tb

            if i % rc_params.displayInterval == 0 and not runtime_error:
                sys.stdout.write("\r%100s\r" % ' ')
                sys.stdout.flush()
                logger.info('[%d/%d][%d/%d] Loss: %f, Cost: %.4fs per batch' %
                            (total_steps, rc_params.niter, i, len(train_loader), loss_avg.val(), eval_time / i))

                if eval_time / i < 0.2:
                    rc_params.displayInterval = 1000
                elif eval_time / i < 0.5:
                    rc_params.displayInterval = 400
                elif eval_time / i < 1.0:
                    rc_params.displayInterval = 200
                prog_bar = mmcv.ProgressBar(rc_params.displayInterval)  # new interval

                # loss_avg.reset()

            # if i % rc_params.valInterval == 0:
            #     val(resnet_crnn, test_dataset, criterion, total_steps, i)
            #     torch.cuda.empty_cache()

        if inf_num >= 10:
            logger.warning("INF loss 10 times!!!")
            # rebuild dataloader and optim

            if rc_params.lr > 0.0001:
                rc_params.lr /= 10
                logger.warning("Reset lr to %f"%rc_params.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = rc_params.lr
            elif rc_params.batchSize > 4:
                rc_params.batchSize = int(rc_params.batchSize/4)
                logger.warning("Divide batch size to %d"%rc_params.batchSize)
                train_loader.batch_sampler.batch_size = rc_params.batchSize
            else:
                rc_params.lr /= 10
                logger.warning("Reset lr to %f"%rc_params.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = rc_params.lr

            logger.warning("Jump Epoch %d"%total_steps)
            continue



        if (total_steps + 1) % rc_params.saveInterval == 0:
            string = "model save to {0}crnn_Rec_done_epoch_{1}.pth".format(log_dir, total_steps)
            logger.info(string)
            torch.save(resnet_crnn.state_dict(), '{0}crnn_Rec_done_epoch_{1}.pth'.format(log_dir, total_steps))

        runtime_error = False
        try:
            torch.cuda.empty_cache()
            val(resnet_crnn, test_dataset, criterion, total_steps, i)
        except RuntimeError as e:
            logger.error(e)
            runtime_error = True
        except ConnectionRefusedError as e:
            logger.error(e)
            runtime_error = True
        finally:
            if runtime_error:
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    val(resnet_crnn, test_dataset, criterion, total_steps, i)
                except RuntimeError as e:
                    logger.error(e)
                    logger.error("Skip Valid")
                except ConnectionRefusedError as e:
                    logger.error(e)
                    logger.error("Skip Valid")





if __name__ == '__main__':

    manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
    log_dir = get_log_dir()
    logger = get_logger(log_dir, rc_params.name, rc_params.name + '_info.log')
    logger.info(opt)

    # tensorboardX
    writer = SummaryWriter(os.path.join(log_dir, 'tb_logs'))

    # store model path
    if not os.path.exists('./expr'):
        os.mkdir('./expr')
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
    # read train set
    tr_t = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.1, hue=0.1)
    ])
    train_dataset = dataset.lmdbDataset(root=opt.trainroot, rgb=True, transform=tr_t, rand_hcrop=True)
    assert train_dataset
    if rc_params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, rc_params.batchSize)
    else:
        sampler = None

    # images will be resize to 32*160
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=rc_params.batchSize,
        shuffle=False, sampler=sampler,
        num_workers=int(rc_params.workers),
        collate_fn=dataset.alignCollate(
            imgH=rc_params.imgH,
            imgW=rc_params.imgW,
            keep_ratio=rc_params.keep_ratio,
            rgb=True)
    )

    # read test set
    # images will be resize to 32*160
    test_dataset = dataset.lmdbDataset(
        root=opt.valroot, rgb=True)

    nclass = len(rc_params.alphabet) + 1
    nc = 1

    converter = utils.strLabelConverter(rc_params.alphabet)
    # criterion = CTCLoss(size_average=True, length_average=True)
    criterion = CTCLoss(size_average=True)

    # cnn and rnn
    image = torch.FloatTensor(rc_params.batchSize, 3, rc_params.imgH, rc_params.imgH)
    text = torch.IntTensor(rc_params.batchSize * 5)
    length = torch.IntTensor(rc_params.batchSize)

    resnet_crnn = ResNetCRNN(rc_params.imgH, nc, nclass, rc_params.nh,
                             resnet_type=rc_params.resnet_type,
                             feat_size=rc_params.feat_size)
    resnet_crnn = torch.nn.DataParallel(resnet_crnn)
    if opt.cuda:
        resnet_crnn.cuda()
        image = image.cuda()
        criterion = criterion.cuda()

    resnet_crnn.apply(weights_init)
    if rc_params.resnet_crnn != '':
        logger.info('loading pretrained model from %s' % rc_params.resnet_crnn)
        state_dict = torch.load(rc_params.resnet_crnn)
        # state_dict_crnn_or = torch.load('../crnn_chinese_characters_rec_backups/work_dirs/extend_dim__lr_0.0001000_batchSize_5_time_0314081842_/crnn_Rec_done_epoch_1.pth')
        if rc_params.resnet_type=='resnet18' and rc_params.feat_size == (1024, 1, 16):
            own_state = resnet_crnn.state_dict()
            for name, param in state_dict.items():
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except RuntimeError as err:
                    if own_state[name].size() == state_dict_crnn_or[name[7:]].size():
                        own_state[name].copy_(state_dict_crnn_or[name[7:]])
                    else:
                        print(own_state[name].size(), state_dict_crnn_or[name[7:]].size())
                        logger.warning("'%s' weight has different weight"%name)
        else:
            resnet_crnn.load_state_dict(state_dict, strict=False)

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if rc_params.sgd:
        optimizer = optim.SGD(resnet_crnn.parameters(), lr=rc_params.lr)
    elif rc_params.adam:
        optimizer = optim.Adam(resnet_crnn.parameters(), lr=rc_params.lr,
                               betas=(rc_params.beta1, 0.999))
    elif rc_params.adadelta:
        optimizer = optim.Adadelta(resnet_crnn.parameters(), lr=rc_params.lr)
    else:
        optimizer = optim.RMSprop(resnet_crnn.parameters(), lr=rc_params.lr)
    # optimizer = lr_scheduler.StepLR(optimizer, step_size=25*len(train_loader), gamma=0.1)

    training()
