dir1 = './to_lmdb/val_label_new.txt'
dir2 = './work_dirs/pretrain_cont_hard_train_lr_0.0001000_batchSize_5_time_0315153209_/epoch_3_step_9258_data.txt'
f1 = open(dir1, 'r')
f2 = open(dir2, 'r')
line1 = f1.readline()
line2 = f2.readline()
r=0
while line1:
    if len(line1.split(' ')[1])==len(line2.split(' ')[1]):
        r+=1
    line1 = f1.readline()
    line2 = f2.readline()    
print(r)        