import alphabets

name = 'base_rms_imgh128f_nl_cont_13'
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
crnn = './history/base_rms_imgh128_lr_0.00010_batchSize_4_time_0306103121_/crnn_Rec_done_epoch_13.pth' #'./work_dirs/base_rms_imgh128f_nl_lr_0.00100_batchSize_16_time_0312143409_/crnn_Rec_done_epoch_2.pth'
beta1 =0.5
lr = 0.0001
niter = 1000
nh = 256
imgW = 10
imgH = 128
batchSize = 7
workers = 2
total_num = 31409
without_fully = False
valInterval = int(60000/batchSize)

