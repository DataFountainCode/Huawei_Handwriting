from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pickle
image_dir = '/home/chenriquan/Datasets/hanzishibie/test_dataset/'
output_dir = '/home/chenriquan/Datasets/hanzishibie/test_visual/'
txt_dir = './work_dirs/with_8_coo_cont_520190308_082830/results/20190308_155009/data/'
    
for i in range(1, 10001):
    img = cv2.imread(image_dir+str(i).zfill(6)+'.png')
    f1 = open(txt_dir+str(i).zfill(6)+'.txt', 'r')
    line  = f1.readline()
    while line:
        line = line.split(' ')
        pred = [int(float(l))for l in line[4:8]]
        cv2.rectangle(img, (pred[0], pred[1]), (pred[2], pred[3]), [0, 255, 0], thickness=1)  
        line  = f1.readline()
    cv2.imwrite(output_dir+str(i).zfill(6)+'.jpg', img)     