import pandas
import numpy as np
import cv2
def IOU(size1, size2):
    U = (size1[2]-size1[0])*(size1[3]-size1[1])+(size2[2]-size2[0])*(size2[3]-size2[1])       
    I = max(min(size1[2], size2[2])-max(size1[0], size2[0]), 0)*max(min(size1[3], size2[3])-max(size1[1], size2[1]), 0)
    return I/(U-I)  
    
csv_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/verify_label.csv'
f = pandas.read_csv(csv_dir, encoding='gbk')
f['x_ul'] = np.min([f['x1'].values, f['x4'].values], 0) 
f['y_ul'] = np.min([f['y1'].values, f['y2'].values], 0)
f['x_br'] = np.max([f['x2'].values, f['x3'].values], 0)
f['y_br'] = np.max([f['y3'].values, f['y4'].values], 0)
d = {}
r = 50005
for i in range(f.shape[0]):
    if i==0:
        s = f.loc[i, 'FileName'][16:21]
        d[s] = str(r).zfill(6)
        r+=1
    elif s != f.loc[i, 'FileName'][16:21]:
        s = f.loc[i, 'FileName'][16:21]  
        d[s] = str(r).zfill(6)   
        r+=1
val_image_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/train_image/'        
pred_dir='./work_dirs/with_8_coo_cont_5_320190309_194625/results/20190310_074210/data/'
save_val_image_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_crop_image_8_coo_all/'
save_val_image_label_dir = '/home/chenriquan/Datasets/hanzishibie/traindataset/val_crop_image_label_8_coo_all/'
num = 0
f['x1_pred'], f['y1_pred'], f['x2_pred'], f['y2_pred'], f['find_pred'] = 0, 0, 0, 0, False
f = f[28273:]
f = f.reset_index(drop=True)
for i in range(f.shape[0]):
    if i==0 :
        s = f.loc[i, 'FileName'][16:21]
        #image_name = d[s]+'.png'
        #img = cv2.imread(val_image_dir+image_name, cv2.IMREAD_GRAYSCALE)  
        #if img is not None:
        #    num+=1
        gt = f.loc[i, 'x_ul':'y_br'].values
        print(d[s])
        _file = open(pred_dir+d[s]+".txt","r")  
        line  = _file.readline()
        preds = []
        while line:    
            #f2.write(line)   
            preds.append([ float(l) for l in line.split(' ')[4:8]])         
            line = _file.readline()  
        _file.close() 
    elif s == f.loc[i, 'FileName'][16:21]:  
        gt = f.loc[i, 'x_ul':'y_br'].values
    else:
        s = f.loc[i, 'FileName'][16:21]
        #image_name = d[s]+'.png'
        #img = cv2.imread(val_image_dir+image_name, cv2.IMREAD_GRAYSCALE)  
        gt = f.loc[i, 'x_ul':'y_br'].values
        _file = open(pred_dir+d[s]+".txt","r")  
        line  = _file.readline()
        preds = []
        while line:    
            #f2.write(line)   
            preds.append([ float(l) for l in line.split(' ')[4:8]])         
            line = _file.readline()  
        _file.close()  
        #if img is not None:
        #    num+=1      
    max_iou = 0         
    for pred in preds:
        _iou = IOU(gt, pred)
        if _iou>0.5:
            f.loc[i, 'find_pred'] = True 
        if _iou>=max_iou:
            f.loc[i, 'x1_pred':'y2_pred'] = pred 
            max_iou = _iou   
print('num: ', num)   
f = f.loc[f['find_pred'], :]   
f = f.reset_index(drop=True)        
f['ratio'] = (f['y2_pred'] - f['y1_pred']) / (f['x2_pred'] - f['x1_pred']) 
ind = np.argsort(f['ratio'].values)
f = f.loc[ind, :]
f = f.reset_index(drop=True)
print(f)
_file = open(save_val_image_label_dir+'val_label.txt', 'a')
r = 1
for i in range(f.shape[0]):
    if i==0:
        s = f.loc[i, 'FileName'][16:21]
        image_name = d[s]+'.png'
        img = cv2.imread(val_image_dir+image_name, cv2.IMREAD_GRAYSCALE)  
        if img is not None:
            num+=1
        else:
            continue             
    elif s != f.loc[i, 'FileName'][16:21]:
        s = f.loc[i, 'FileName'][16:21]
        image_name = d[s]+'.png'
        img = cv2.imread(val_image_dir+image_name, cv2.IMREAD_GRAYSCALE)  
        if img is not None:
            num+=1    
        else:
            continue      
    crop_img = img[int(f.loc[i, 'y1_pred']): int(f.loc[i, 'y2_pred']), int(f.loc[i, 'x1_pred']): int(f.loc[i, 'x2_pred'])] 
    crop_img = np.rot90(crop_img)
    crop_image_name = str(r).zfill(6)+'.jpg'
    cv2.imwrite(save_val_image_dir+crop_image_name, crop_img)
    _file.write(save_val_image_dir+crop_image_name+' %s\n'%(f.loc[i, 'text']))
    r += 1
_file.close()        
print('num: ', num)
print('r: ', r)  










                     
