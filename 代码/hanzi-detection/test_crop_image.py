import cv2
import numpy as np
import pickle
image_dir = '/home/chenriquan/Datasets/hanzishibie/test_dataset/'
image_save_dir = '/home/chenriquan/Datasets/hanzishibie/test_crop_image/'
label_dir = '/home/chenriquan/Datasets/hanzishibie/test_crop_label/'
pkl_dir = '/home/chenriquan/Datasets/hanzishibie/test_dict/'
output_dir = './work_dirs/test_out_20190314_083532/results/20190314_083543/data/'
r=1
d = {}
f2 = open(label_dir+'test_index.txt', 'a')
for i in range(1, 10001):
    #print(image_dir+str(i).zfill(6)+'.png')
    img = cv2.imread(image_dir+str(i).zfill(6)+'.png', cv2.IMREAD_GRAYSCALE) 
    if img is None:
        print('read error')
        break
    f = open(output_dir+str(i).zfill(6)+'.txt', "r")
    line  = f.readline()
    while line:    
        #f2.write(line)   
        preds = [ int(float(l)) for l in line.split(' ')[4:8]]
        
        crop_img = img[preds[1]:preds[3], preds[0]:preds[2]]     
        crop_img = np.rot90(crop_img)
        '''
        print('img_shape: ', img.shape)
        print('crop_shape: ', crop_img.shape)
        print('coo: ', preds[1], preds[3], preds[0], preds[2])
        print('r: ', r, 'crop_shape', crop_img.shape)
        '''
        if (crop_img.shape[0]==0) or (crop_img.shape[1]==0) :
            line = f.readline() 
            continue
        crop_image_name = str(r).zfill(6)+'.jpg'
        cv2.imwrite(image_save_dir+crop_image_name, crop_img)
        f2.write(image_save_dir+crop_image_name+' 不\n')    
        d[str(r).zfill(6)] = [str(i).zfill(6)+'.png', preds]
        line = f.readline()  
        r += 1   
    f.close() 
f2.close()    
with open(pkl_dir + 'test_dict' + '.pkl', 'wb') as f:
    pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)