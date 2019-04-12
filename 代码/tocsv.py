# coding=gbk
import csv
import pickle
pred_dir = './pred/pred.txt'
pred_csv_dir = './pred_csv/predict.csv'
f1 = open('pred_dict.pkl', 'rb')
d = pickle.load(f1)
f = open(pred_dir, 'r')
line = f.readline()
out = open(pred_csv_dir,'a', newline='')
csv_write = csv.writer(out,dialect='excel')
csv_write.writerow(['filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'text'])
while line:
    line = line.split(' ')
    line[9] = line[9].strip()
    line[0] = d[line[0][0:6]]
    csv_write.writerow(line)
    line = f.readline()
print ("write over")