import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 3, 2]                    #            3, 3, 3, 3, 3, 3, 2, 2, 2
        ps = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 128, 256, 256, 512, 512, 1024, 1024]  #    64, 128, 128, 256, 256, 256, 512, 512, 512       64, 128, 256, 256, 512, 512, 512, 512, 512
        if imgH==64:
            pass # ps[-2]=0 #pass# 
        if imgH == 128:
            ss[-1] = (2, 1)

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)   # 3x32x128
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 128x8x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(6, True)
        convRelu(7)
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(8, True)  # 512x1x16
        if imgH == 128:
            cnn.add_module('pooling{0}'.format(5),
                           nn.MaxPool2d((2, 2), (2, 1), (0, 0)))  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input, cp=False):
        # conv features

        if not cp:
            conv = self.cnn(input)
            b, c, h, w = conv.size()
            assert h == 1, "the height of conv must be 1"
            conv = conv.squeeze(2) # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # rnn features
            output = self.rnn(conv)
        else:
            b = input.size(0)
            conv = nn.Sequential(*self.cnn[0:3])(input)
            conv = checkpoint_sequential(nn.Sequential(*self.cnn[3:]), 7, conv).view(b, 512, -1)
            # b, c, h, w = conv.size()
            # assert h == 1, "the height of conv must be 1"
            # conv = conv.squeeze(2) # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # rnn features
            output = checkpoint_sequential(self.rnn, 2, conv)


        return output
