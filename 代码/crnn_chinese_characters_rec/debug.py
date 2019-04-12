from warpctc_pytorch import CTCLoss
from models.resnet_crnn import ResNetCRNN
from models.crnn import CRNN
import torch
from torch.utils.checkpoint import checkpoint
from time import time

r = CRNN(128, 3, 9100, 512).cuda()
input = torch.ones(size=(4, 3, 128, 300000)).cuda()
c = CTCLoss().cuda()
# input.requires_grad =True

cp=True
try:
    ok = False
    try:
        start = time()
        output = r(input, cp=cp)
        l = torch.ones(size=(8, 30)).int()
        lsize=torch.IntTensor([30]*8)
        psize=torch.IntTensor([output.size(0)]*output.size(1))
        if cp:
            loss = checkpoint(c, output, l.view(-1), psize, lsize)
        else:
            loss = c(output, l.view(-1), psize, lsize)
        loss.backward()
        print('cost', time()-start)
        ok = True
    except:
        raise RuntimeError("RE")
    finally:
        if ok:
            pass
        else:
            raise RuntimeError("FE")
except RuntimeError as e:
    print("Error", e)
    import pdb; pdb.set_trace()

print('success')
