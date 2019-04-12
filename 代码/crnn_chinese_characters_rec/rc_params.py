import alphabets

resnet_type='resnet18'
feat_size=(512, 1, 16)
name = resnet_type+'_rcnn_sgd_imgh128_rgb_%dx%dx%d'%feat_size
log_dir = './work_dirs'
random_sample = True
keep_ratio = True
sgd = True
adam = False
adadelta = False   # all false to use RMS optim
saveInterval = 1
valInterval = 10000
n_test_disp = 10
displayInterval = 100
tbInterval = 1000
experiment = './expr'
alphabet = alphabets.alphabet
# F1 15 ./work_dirs/resnet18_rcnn_rms_imgh128_rgb_1x3_old_traindata_lr_0.00100_batchSize_8_time_0314162700_/crnn_Rec_done_epoch_1.pth
# resnet18 F1 77 ./work_dirs/resnet18_rcnn_adam_imgh128_rgb_1x16_old_traindata_lr_0.00010_batchSize_8_time_0315161501_/crnn_Rec_done_epoch_2.pth
# resnet18 F1 79 ./work_dirs/resnet18_rcnn_adam_imgh128_rgb_1x16_old_traindata_lr_0.00010_batchSize_4_time_0316005937_/crnn_Rec_done_epoch_2.pth
# ./work_dirs/resnet18_rcnn_rms_imgh128_rgb_1x3_old_traindata_lr_0.00010_batchSize_4_time_0315104506_/crnn_Rec_done_epoch_0.pth
resnet_crnn = './work_dirs/resnet18_rcnn_sgd_imgh128_rgb_512x1x16_lr_0.00100_batchSize_8_time_0319110013_/crnn_Rec_done_epoch_7.pth'
beta1 =0.5
lr = 0.001
niter = 1000
nh = feat_size[0] // 2
imgW = 128
imgH = 128
batchSize = 4
workers = 4
total_num = 31409

valInterval = int(60000/batchSize)
