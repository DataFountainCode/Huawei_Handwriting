import alphabets

name = 'test'
log_dir = './work_dirs'
random_sample = True
keep_ratio = True
adam = False
adadelta = False   # all false to use RMS optim

rgb=False

saveInterval = 1
valInterval = 10000
n_test_disp = 10
displayInterval = 10000
tbInterval = 10000
experiment = './expr'
alphabet = alphabets.alphabet
crnn =  './work_dirs/test/crnn_Rec_done_epoch_3.pth'# './work_dirs/test_lr_0.00010_batchSize_5_time_0312154131_/crnn_Rec_done_epoch_1.pth'#'./work_dirs/rms_cont_46_7_3_1_14_concat_train_lr_0.00010_batchSize_5_time_0311211237_/crnn_Rec_done_epoch_8.pth'
beta1 =0.5
lr = 0.00003
niter = 1000
nh = 256
imgW = 10
imgH = 64
batchSize = 7
workers = 2
total_num = 31409
without_fully = False
valInterval = int(60000/batchSize)

