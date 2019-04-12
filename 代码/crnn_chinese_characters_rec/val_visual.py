from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pickle
image_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/train_image/'
dict_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_visual_dict/'
val_visual_image = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_visual_image/'
label = './to_lmdb/val_visual.txt'
pred = './work_dirs/base_rms_visual_lr_0.00010_batchSize_1_time_0308094624_/epoch_0_step_0_data.txt'
f = open(dict_dir+'d1_dict.pkl', 'rb')
d1 = pickle.load(f)
f.close()
f = open(dict_dir+'d2_dict.pkl', 'rb')
d2 = pickle.load(f)
f.close()
f = open(label, 'r')
labels = f.readlines()
f.close()
f = open(pred, 'r')
preds = f.readlines()
f.close()
for i in range(50005, 60001):
    img = cv2.imread(image_dir+str(i).zfill(6)+'.png') 
    if img is None:
        print('failed find image')
        break
    if str(i).zfill(6) in d1.keys():
        for box in d1[str(i).zfill(6)]:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 0], thickness=1)  
    if str(i).zfill(6) in d2.keys():  
        for box in d2[str(i).zfill(6)]:
            _label = labels[int(box[0])-1].split(' ')[1]
            _pred =  preds[int(box[0])-1].split(' ')[1]
            if _label!=_pred:
                cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), [0, 0, 255], thickness=1) 
                font = ImageFont.truetype('msyh.ttf', 10)
                img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                fillColor = (255,0,0)
                down = 0
                draw = ImageDraw.Draw(img_PIL)  
                for k,s2 in enumerate(_pred.strip()):            
                    if k == 0:
                        w,h = font.getsize(s2)   #获取第一个文字的宽和高
                    else:
                        down = down+h         
                    draw.text((int(box[1]), int(box[2])+down), s2, (255,0,0), font=font)
                img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR) 
    print(i, cv2.imwrite(val_visual_image+str(i).zfill(6)+'.jpg', img))               
         