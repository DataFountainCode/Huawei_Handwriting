import numpy as np
import pandas as pd
import mmcv
gt_dir = './to_lmdb/val_label_new.txt'
pred_dir = './history/test_out_lr_0.00010_batchSize_1_time_0310105604_/epoch_0_step_0_data.txt'
stat_csv = './False_record/stat.csv'
f1 = open(gt_dir, 'r')
f2 = open(pred_dir, 'r')
d={}
prog_bar = mmcv.ProgressBar(31402)
for i in range(31402):
    pred = f2.readline().split(' ')[1].strip()
    gt = f1.readline().split(' ')[1].strip()
    for p in gt:
        if p not in pred:
            if p in d.keys():
                d[p]+=1
            else:
                d[p]=1  
    prog_bar.update()  
p=pd.DataFrame(columns=['word', 'num'])
for key in d.keys():
    p=p.append(pd.DataFrame([[key, d[key]]], 
                                 columns=['word', 'num']))
print(p)
p = p.reset_index(drop=True)  
ind = np.argsort(p['num'].values, )
p = p.loc[ind[::-1], :]
p = p.reset_index(drop=True)  
print(p)  
p.to_csv(stat_csv)                              
                    
