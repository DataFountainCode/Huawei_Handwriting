import cv2
import pandas as pd
import numpy as np
import pickle 
image_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/train_image/'
image_label = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_label/'
pred_dir = './work_dirs/faster_rcnn_r50_fpn_hanzishibie_8_coo_20190223_144150/results/20190225_232441/data/'
crop_image_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_crop_image_8_coo_visual/'
crop_image_label = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_crop_image_label_8_coo_visual/'
val_dict = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_visual_dict/'
def IOU(size1, size2):
    U = (size1[2]-size1[0])*(size1[3]-size1[1])+(size2[2]-size2[0])*(size2[3]-size2[1])       
    I = max(min(size1[2], size2[2])-max(size1[0], size2[0]), 0)*max(min(size1[3], size2[3])-max(size1[1], size2[1]), 0)
    return I/(U-I)  
p = pd.DataFrame(columns=['name', 'label', 'x1', 'y1', 'x2', 'y2'])
d1 = {}
for i in range(50005, 60001):
    f1 = open(pred_dir+str(i).zfill(6)+".txt","r")  
    line  = f1.readline()
    preds = []
    while line:    
        #f2.write(line)   
        preds.append([ float(l) for l in line.split(' ')[4:8]])         
        line = f1.readline()  
    f1.close() 
    f1 = open(image_label+str(i).zfill(6)+".txt","r")  
    line  = f1.readline()
    gts = []
    while line:    
        #f2.write(line)   
        box = [ float(l) for l in line.split(' ')[0:4]]
        box.append(line.split(' ')[4].strip())
        gts.append(box)         
        line = f1.readline()  
    f1.close() 
    for pred in preds:
        with_gt = False
        for gt in gts:
            if IOU(pred, gt[:4])>0.5:
                p=p.append(pd.DataFrame([[str(i).zfill(6), gt[4], pred[0], pred[1], pred[2], pred[3]]], columns=['name', 'label', 'x1', 'y1', 'x2', 'y2']))
                with_gt = True
                break
        if with_gt==False:
            if str(i).zfill(6) in d1.keys():
                d1[str(i).zfill(6)].append([pred[0], pred[1], pred[2], pred[3]])  
            else:
                d1[str(i).zfill(6)]=[]
                d1[str(i).zfill(6)].append([pred[0], pred[1], pred[2], pred[3]])    

p = p.reset_index(drop=True) 
p['ratio'] = (p['y2']-p['y1'])/(p['x2']-p['x1'])
ind = np.argsort(p['ratio'].values)
p = p.loc[ind, :]
p = p.reset_index(drop=True)
print(p)
f = open(crop_image_label+'val_visual.txt', 'a')
d2={}


for i in range(1, p.shape[0]+1):
    img = cv2.imread(image_dir+p.loc[i-1, 'name']+'.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('failed find image')
        break
    crop_img = img[int(p.loc[i-1, 'y1']): int(p.loc[i-1, 'y2']), int(p.loc[i-1, 'x1']): int(p.loc[i-1, 'x2'])] 
    crop_img = np.rot90(crop_img)
    crop_image_name = str(i).zfill(6)+'.jpg'
    print(crop_img.shape)
    print(cv2.imwrite(crop_image_dir+crop_image_name, crop_img))
    f.write(crop_image_dir+crop_image_name+' %s\n'%(p.loc[i-1, 'label']))  
    if p.loc[i-1, 'name'] in d2.keys():
        d2[p.loc[i-1, 'name']].append(str(i).zfill(6))
    else:
        d2[p.loc[i-1, 'name']]=[]
        d2[p.loc[i-1, 'name']].append([str(i).zfill(6), int(p.loc[i-1, 'x1']), int(p.loc[i-1, 'y1']), int(p.loc[i-1, 'x2']), int(p.loc[i-1, 'y2'])])  

                    
with open(val_dict + 'd1_dict' + '.pkl', 'wb') as f:
    pickle.dump(d1, f, pickle.HIGHEST_PROTOCOL)

with open(val_dict + 'd2_dict' + '.pkl', 'wb') as f:
    pickle.dump(d2, f, pickle.HIGHEST_PROTOCOL)



                      