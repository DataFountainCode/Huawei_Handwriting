import torch
from torch.autograd import Variable
x=Variable(torch.tensor([2.0]), requires_grad=True)
y=Variable(torch.tensor([2.0]), requires_grad=True)
w=x+y
w1=w.data
w2=w+x
w2.backward()
print(x.grad)