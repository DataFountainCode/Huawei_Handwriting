import pandas 
import numpy as np
import cv2
f = pandas.read_csv('train_label.csv', encoding='gbk')
f = f.reset_index(drop=True)
f['x_ul'] = np.min([f['x1'].values, f['x4'].values], 0) 
f['y_ul'] = np.min([f['y1'].values, f['y2'].values], 0)
f['x_br'] = np.max([f['x2'].values, f['x3'].values], 0)
f['y_br'] = np.max([f['y3'].values, f['y4'].values], 0)
f['ratio'] = (f['y_br'] - f['y_ul']) / (f['x_br'] - f['x_ul']) 
ind = np.argsort(f['ratio'].values)
r=1

d = {}
for i in range(f.shape[0]):
    if i==0:
        s = f.loc[i, 'FileName'][16:21]
        d[s] = str(r).zfill(6)
        r+=1
    elif s != f.loc[i, 'FileName'][16:21]:
        s = f.loc[i, 'FileName'][16:21]  
        d[s] = str(r).zfill(6)   
        r+=1
                
f = f.loc[ind, :]
f = f.reset_index(drop=True)
r = 1
num = 0
image_dir="/home/chenriquan/Datasets/hanzishibie/traindataset/train_crop_image_all/"
print('row_num: ', f.shape[0])
_file = open('./train_crop_image_label_all/train_label.txt', 'a')
for i in range(f.shape[0]):
    s = f.loc[i, 'FileName'][16:21]
    image_name = d[s]+'.png'
    img = cv2.imread('./train_image/'+image_name, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        num+=1
    if (f.loc[i, 'y_ul']<f.loc[i, 'y_br'])and(f.loc[i, 'x_ul']<f.loc[i, 'x_br']):    
        crop_img = img[f.loc[i, 'y_ul']: f.loc[i, 'y_br'], f.loc[i, 'x_ul']: f.loc[i, 'x_br']] 
        crop_img = np.rot90(crop_img)
        crop_image_name = str(r).zfill(6)+'.jpg'
        cv2.imwrite(image_dir+crop_image_name, crop_img)
        _file.write(image_dir+crop_image_name+' %s\n'%(f.loc[i, 'text']))
        r +=1
_file.close()        
print('num: ', num)
print('r: ', r)        
        
      
      
    
