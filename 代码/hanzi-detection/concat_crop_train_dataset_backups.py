import cv2
import pandas as pd
import numpy as np
import mmcv
print(1)
image_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/train_image/'
image_label_csv = '/home/chenriquan/Datasets/hanzishibie/traindataset/train_label.csv'
save_image = '/home/chenriquan/Datasets/hanzishibie/traindataset/train_crop_image_new/'
save_label = '/home/chenriquan/Datasets/hanzishibie/traindataset/train_crop_image_new_label/train_new_label.txt'
detetction_pred = './work_dirs/train_out_20190310_152758/results/20190310_152810/data/'
def IOU(size1, size2):
    U = (size1[2]-size1[0])*(size1[3]-size1[1])+(size2[2]-size2[0])*(size2[3]-size2[1])       
    I = max(min(size1[2], size2[2])-max(size1[0], size2[0]), 0)*max(min(size1[3], size2[3])-max(size1[1], size2[1]), 0)
    return I/(U-I)  
d = {}
f1 = pd.DataFrame(columns=['name', 'label', 'x1', 'y1', 'x2', 'y2'])
f2 = pd.read_csv(image_label_csv, encoding='gbk')
f2['x_ul'] = np.min([f2['x1'].values, f2['x4'].values], 0) 
f2['y_ul'] = np.min([f2['y1'].values, f2['y2'].values], 0)
f2['x_br'] = np.max([f2['x2'].values, f2['x3'].values], 0)
f2['y_br'] = np.max([f2['y3'].values, f2['y4'].values], 0)
r = 1
prog_bar = mmcv.ProgressBar(f2.shape[0])
for i in range(f2.shape[0]):
    if i==0:
        s = f2.loc[i, 'FileName'][16:21]
    elif s != f2.loc[i, 'FileName'][16:21]:
        s = f2.loc[i, 'FileName'][16:21]
        r += 1
    if str(r).zfill(6) in d.keys():
        d[str(r).zfill(6)].append([f2.loc[i, 'text'], f2.loc[i, 'x_ul'], f2.loc[i, 'y_ul'], f2.loc[i, 'x_br'],f2.loc[i, 'y_br']])   
    else:
        d[str(r).zfill(6)] = []
        d[str(r).zfill(6)].append([f2.loc[i, 'text'], f2.loc[i, 'x_ul'], f2.loc[i, 'y_ul'], f2.loc[i, 'x_br'],f2.loc[i, 'y_br']])   
    prog_bar.update() 
print(f1)
d1 = []
prog_bar = mmcv.ProgressBar(50004)
for i in range(1, 50005):
    f3 = open(detetction_pred+str(i).zfill(6)+'.txt', 'r') 
    line = f3.readline()     
    preds = []
    while line:    
        #f2.write(line)   
        preds.append([ float(l) for l in line.split(' ')[4:8]])         
        line = f3.readline()  
    f3.close()   
    gts = d[str(i).zfill(6)]
    
    for gt in gts:
        max_iou=0
        find = False
        for pred in preds:
            _iou = IOU(gt[1:], pred)
            if _iou>0.7 and _iou>max_iou:
                d_back = dict(name = str(i).zfill(6), label = gt[0], x1 = pred[0], y1 = pred[1], x2 = pred[2], y2 = pred[3])
                max_iou = _iou
                find = True
        if find:        
            d1.append(d_back)
    prog_bar.update() 
f1 = pd.DataFrame(d1)                
f1 = f1.reset_index(drop=True) 
f1['ratio'] = (f1['y2']-f1['y1'])/(f1['x2']-f1['x1'])
ind = np.argsort(f1['ratio'].values)
f1 = f1.loc[ind, :]
f1 = f1.reset_index(drop=True)  
print(f1)
_file = open(save_label, 'a')
r = 1
prog_bar = mmcv.ProgressBar(f1.shape[0])
for i in range(f1.shape[0]):
    if i==0:
        s = f1.loc[i, 'name']
        image_name = s+'.png'
        img = cv2.imread(image_dir+image_name, cv2.IMREAD_GRAYSCALE)  
        if img is not None:
            pass
        else:
            print('failed to find image: ', image_dir+image_name)
            break             
    elif s != f1.loc[i, 'name']:
        s = f1.loc[i, 'name']
        image_name = s+'.png'
        img = cv2.imread(image_dir+image_name, cv2.IMREAD_GRAYSCALE)  
        if img is not None:
            pass
        else:
            print('failed to find image: ', image_dir+image_name)
            break       
    crop_img = img[int(f1.loc[i, 'y1']): int(f1.loc[i, 'y2']), int(f1.loc[i, 'x1']): int(f1.loc[i, 'x2'])] 
    crop_img = np.rot90(crop_img)
    crop_image_name = str(r).zfill(6)+'.jpg'
    cv2.imwrite(save_image+crop_image_name, crop_img)
    _file.write(save_image+crop_image_name+' %s\n'%(f1.loc[i, 'label']))
    r += 1
    prog_bar.update() 
_file.close() 























