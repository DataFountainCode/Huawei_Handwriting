#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np


class lmdbDataset(Dataset):

    def __init__(self, root=None, rgb=False, transform=None, target_transform=None, rand_hcrop=False):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:

            str = 'num-samples'
            nSamples = int(txn.get(str.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.rand_hcrop = rand_hcrop
        self.target_transform = target_transform
        self.rand_crop_r = 0.1
        self.rgb = rgb

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.rgb:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
                
            if self.rand_hcrop: # random crop image is rotated
                w, h = img.size
                if np.random.rand() >= 0.5:
                    h = int((1-self.rand_crop_r*np.random.rand())*h)
                    img = img.crop((0, 0, w, h))
                else:
                    lh = int(self.rand_crop_r*np.random.rand()*h)
                    img = img.crop((0, lh, w, h))
            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode())

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class lmdbDataset3(Dataset):

    def __init__(self, root=None, mode='up', transform=None, target_transform=None, rand_hcrop=False):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:

            str = 'num-samples'
            nSamples = int(txn.get(str.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.rand_hcrop = rand_hcrop
        self.target_transform = target_transform
        self.rand_crop_r = 0.1
        self.mode = mode

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
                
            if self.rand_hcrop: # random crop image is rotated
                w, h = img.size
                print('w, h: ', w, h)
                if np.random.rand() >= 0.5:
                    h = int((1-self.rand_crop_r*np.random.rand())*h)
                    img = img.crop((0, 0, w, h))
                else:
                    lh = int(self.rand_crop_r*np.random.rand()*h)
                    img = img.crop((0, lh, w, h))
            if self.mode == 'up':
                w, h = img.size
                img = img.resize((w, int(h*1.2)),Image.ANTIALIAS)
                img = img.crop((0, 0, w, h))
            if self.mode == 'mid':
                w, h = img.size
                img = img.resize((w, int(h*1.2)),Image.ANTIALIAS)
                img = img.crop((int(h*0.1), 0, w, int(h*0.1)+h))   
            if self.mode == 'down':
                w, h = img.size
                img = img.resize((w, int(h*1.2)),Image.ANTIALIAS)
                img = img.crop((int(h*0.2), 0, w, int(h*0.2)+h))         
            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode())

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR, rgb=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.rgbNormalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.rgb=rgb


    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        if self.rgb:
            img = rgbNormalizer(img)
        else:
            img.sub_(0.5).div_(0.5)
        return img


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR, rgb=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.rgbNormalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.rgb=rgb


    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        if self.rgb:
            img = rgbNormalizer(img)
        else:
            img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        batch_list = [i for i in range(0, n_batch)]
        for i in range(n_batch):
            random_ind = random.randint(0, len(batch_list)-1)
            random_start = batch_list[random_ind]
            del batch_list[random_ind]
            batch_index = random_start*self.batch_size + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            tail_index = self.batch_size*n_batch + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index
        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=256, keep_ratio=False, min_ratio=1, rgb = False, with_resize=False, ratio_ranges=None):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.with_resize = with_resize
        self.ratio_range = ratio_ranges
        self.rgb = rgb

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            if self.with_resize:
                rand = (np.random.rand()*(self.ratio_range[1]-self.ratio_range[0])+self.ratio_range[0])
            for image in images:
                w, h = image.size
                if self.with_resize:
                    ratios.append(w / float(h)*rand)
                else:
                    ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH), self.rgb)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

