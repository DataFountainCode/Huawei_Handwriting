from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pickle
image_dir = '/home/chenriquan/Datasets/hanzishibie/test_dataset/'
output_dir = '/home/chenriquan/Datasets/hanzishibie/test_visual/'
txt_dir = './work_dirs/resnet18_rcnn_sgd_imgh128_rgb_512x1x16_lr_0.00100_batchSize_1_time_0322162554_/epoch_0_step_1_data.txt'
pred_dir = '/home/chenriquan/Datasets/hanzishibie/pred/'
dict_dir = '/home/chenriquan/Datasets/hanzishibie/test_dict/'

f = open(dict_dir+'test_dict.pkl', 'rb')
d = pickle.load(f)
f1 = open(txt_dir, 'r')
line  = f1.readline()
f2 = open(pred_dir+'pred.txt', 'a')
s=''
while line:
    line = line.split(' ')
    if s=='':
        s = d[line[0][0:6]][0]
        img = cv2.imread(image_dir+s)
    elif s!=d[line[0][0:6]][0]:
        cv2.imwrite(output_dir+s, img)    
        s = d[line[0][0:6]][0]
        img = cv2.imread(image_dir+s)
    pred =  d[line[0][0:6]][1]    
    cv2.rectangle(img, (pred[0], pred[1]), (pred[2], pred[3]), [0, 255, 0], thickness=1)  
    font = ImageFont.truetype('msyh.ttf', 10)
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fillColor = (255,0,0)
    position = (pred[0],pred[1])
    draw = ImageDraw.Draw(img_PIL)  
    draw.text(position, line[1], font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    f2.write(' '.join([s, str(pred[0]), str(pred[1]), str(pred[2]), str(pred[1]), str(pred[2]), str(pred[3]), str(pred[0]), str(pred[3]), line[1]]))
    line = f1.readline()

cv2.imwrite(output_dir+s, img)     
             
        
        
        