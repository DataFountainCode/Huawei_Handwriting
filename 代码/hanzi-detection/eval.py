r=0
import numpy as np

def IOU(size1, size2):
    U = (size1[2]-size1[0])*(size1[3]-size1[1])+(size2[2]-size2[0])*(size2[3]-size2[1])       
    I = max(min(size1[2], size2[2])-max(size1[0], size2[0]), 0)*max(min(size1[3], size2[3])-max(size1[1], size2[1]), 0)
    return I/(U-I)   
correct = 0
total = 0    
for i in range(50005, 60001):
    f = open("/home/chenriquan/Datasets/hanzishibie/traindataset/train_label/"+str(i).zfill(6)+".txt","r")  
    r+=1
    #f2 = open("train_label/"+ str(r).zfill(6) +".txt",'a')
    line  = f.readline()
    gts = []
    while line:    
        #f2.write(line)   
        gts.append([ float(l) for l in line.split(' ')[4:8]])         
        line = f.readline()  
    f.close() 
    f = open("./work_dirs/faster_rcnn_r50_fpn_hanzishibie_base_20190221_221347/results/20190221_221611/data/"+str(i).zfill(6)+".txt","r")  
    line  = f.readline()
    preds = []
    while line:    
        #f2.write(line)   
        preds.append([ float(l) for l in line.split(' ')[4:8]])         
        line = f.readline()  
    f.close() 
    _correct = 0
    _total = len(gts)
    for gt in gts:
        max_iou = 0
        for pred in preds:
            _iou = IOU(gt, pred)
            max_iou = max(_iou, max_iou)
        if max_iou>0.7:
            _correct+=1    
    correct+=_correct
    total+=_total
print(correct, total)    
      