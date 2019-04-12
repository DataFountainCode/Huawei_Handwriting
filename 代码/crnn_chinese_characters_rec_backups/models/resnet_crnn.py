import torch.nn as nn
from .crnn import BidirectionalLSTM
from torchvision.models import resnet50, resnet18
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class ResNetCRNN(nn.Module):
    #                   32    1   37     512
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, resnet_type='resnet18', feat_size=(512, 1, 16)):
        super(ResNetCRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # ks = [3, 3, 3, 3, 3, 3, 3, 3, 2]
        # ps = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 128, 256, 256, 256, 512, 512, 512]
        #
        # cnn = nn.Sequential()
        #
        # def convRelu(i, batchNormalization=False):
        #     nIn = nc if i == 0 else nm[i - 1]
        #     nOut = nm[i]
        #     cnn.add_module('conv{0}'.format(i),
        #                    nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        #     if batchNormalization:
        #         cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        #     if leakyRelu:
        #         cnn.add_module('relu{0}'.format(i),
        #                        nn.LeakyReLU(0.2, inplace=True))
        #     else:
        #         cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        #
        # convRelu(0)
        # cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        # convRelu(1)
        # cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        # convRelu(2, True)
        # convRelu(3)
        # cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 128x8x32
        # convRelu(4, True)
        # convRelu(5)
        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        # convRelu(6, True)
        # convRelu(7)
        # cnn.add_module('pooling{0}'.format(4),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        # convRelu(8, True)  # 512x1x16
        # if imgH == 128:
        #     cnn.add_module('pooling{0}'.format(5),
        #                    nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x1x16

        class identity(nn.Module):
            def forward(self, x):
                return x

        self.resnet_type = resnet_type
        self.feat_size = feat_size
        if self.resnet_type == 'resnet18':
            # resnet18
            self.cnn = resnet18(pretrained=True)

            if self.feat_size[2] == 16:
                self.cnn.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
                self.cnn.layer3[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 1), bias=False)

            self.cnn.layer3.add_module('maxpool', nn.MaxPool2d((2, 2), (2, 1)))

            if self.feat_size[2] == 8 or self.feat_size[2] == 16:
                self.cnn.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1), bias=False)
                self.cnn.layer4[0].downsample[0] = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 1), bias=False)

            if not self.feat_size == (1024, 1, 16):
                self.cnn.layer4.add_module('maxpool', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

            self.cnn.avgpool = identity()
            self.cnn.fc = identity()
            # (1, 3, 128, 128) -self.cnn-> (1, 2048, 1, 3)

            # def hook(self, in_, out_):
            #     print(out_.size())
            # self.cnn.layer4.register_forward_hook(hook)

            self.rnn = nn.Sequential(
                BidirectionalLSTM(self.feat_size[0], nh, nh),
                BidirectionalLSTM(nh, nh, nclass))

        elif self.resnet_type == 'resnet50':
            # resnet50

            self.cnn = resnet50(pretrained=True)
            # self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # self.cnn.layer3[0].conv2 = nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False)
            # self.cnn.layer3[0].downsample[0] = nn.Conv2d(512, 1024, (1, 1), (2, 2), bias=False)
            self.cnn.layer3.add_module('maxpool', nn.MaxPool2d((2, 2), (2, 1), (0, 0)))

            self.cnn.layer4[0].conv2 = nn.Conv2d(512, 512, (3, 3), (2, 1), (1, 1), bias=False)
            self.cnn.layer4[0].downsample[0] = nn.Conv2d(1024, 2048, (1, 1), (2, 1), bias=False)
            self.cnn.layer4.add_module('maxpool', nn.MaxPool2d((2, 2), (2, 2), (0, 0)))

            # def hook(self, in_, out_):
            #     print(out_.size())
            # self.cnn.layer4.register_forward_hook(hook)

            self.cnn.avgpool = identity()
            self.cnn.fc = identity()
            # (1, 3, 128, 128) -self.cnn-> (1, 2048, 1, 3)

            self.rnn = nn.Sequential(
                BidirectionalLSTM(2048, nh, nh),
                BidirectionalLSTM(nh, nh, nclass))

        else:
            raise TypeError("resnet_type is error!!!")

        # for p in self.cnn.conv1.parameters():
        #     p.requires_grad = False
        # for p in self.cnn.bn1.parameters():
        #     p.requires_grad = False
        # for p in self.cnn.layer1.parameters():
        #     p.requires_grad = False
        # for p in self.cnn.layer2.parameters():
        #     p.requires_grad = False

    def forward(self, input, cp=False):
        # conv features
        b = input.size(0)

        if cp == False:
            if self.resnet_type == 'resnet18':
                if self.feat_size == (1024, 1, 16):
                    conv = self.cnn(input).view(b, 1024, -1)
                else:
                    conv = self.cnn(input).view(b, 512, -1)  # for resnet18
                # import pdb; pdb.set_trace()
            elif self.resnet_type == 'resnet50':
                conv = self.cnn(input).view(b, 2048, -1)  # for resnet50
            else:
                raise TypeError("resnet_type is error!!!")
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            output = self.rnn(conv)
            return output
        else:
            # save gpu memory
            conv = self.cnn.conv1(input)
            conv = self.cnn.bn1(conv)
            conv = self.cnn.relu(conv)
            conv = self.cnn.maxpool(conv)

            # conv = checkpoint_sequential(self.cnn.layer1, 3, conv)
            # conv = checkpoint_sequential(self.cnn.layer2, 4, conv)
            conv = self.cnn.layer1(conv)
            conv = self.cnn.layer2(conv)

            conv = self.cnn.layer3[0](conv)
            conv = checkpoint_sequential(nn.Sequential(*self.cnn.layer3[1:]), 5, conv)
            conv = checkpoint_sequential(self.cnn.layer4, 3, conv).view(b, 2048, -1)
            conv = conv.permute(2, 0, 1)
            output = checkpoint_sequential(self.rnn, 2, conv)
            return output

