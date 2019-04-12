import torchvision.transforms as transforms
import cv2
from PIL import Image
import mmcv
import numpy as np

t = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=0, scale=(1.0, 1.01), shear=(0, 0.1), resample=Image.NEAREST)
])

r = 0.2

img = cv2.imread('/home/chenriquan/Datasets/CC/image/img_calligraphy_00003_bg.jpg')
tranno = mmcv.load('/home/chenriquan/Datasets/CC/train_anno.pickle')[2]
q = tranno['ann']['quadrilateral'][2].reshape(4, 2)
img = img[int(q[:, 1].min()):int(q[:, 1].max()), int(q[:, 0].min()):int(q[:, 0].max()), :]

if np.random.rand()>0.5:
    w = int(r*np.random.rand()*img.shape[1])
    img = img[:, w:, :]
else:
    w = int((1-r*np.random.rand())*img.shape[1])
    img = img[:, :w, :]

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
img = t(img)
img.save('demo.jpg')
