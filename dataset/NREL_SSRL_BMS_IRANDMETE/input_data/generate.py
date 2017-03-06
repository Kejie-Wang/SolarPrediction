# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '20/11/2016'

import numpy as np
import sys
import os
import time

HOUR_IN_A_DAY = 24

validation_day_num = 365
test_day_num = 366

if len(sys.argv) != 2:
	print "Error: please input the train and validation set prop, e.g. python generate.py 200011201205"
	exit(0)

data_dir = sys.argv[1] + "/"

#load the data
ir_data = np.loadtxt(data_dir + 'ir_data.csv', delimiter=',')
mete_data = np.loadtxt(data_dir + 'mete_data.csv', delimiter=',')
target_data = np.loadtxt(data_dir + 'target_data.csv', delimiter=',')

data_hour_length = len(ir_data)
validation_length = validation_day_num * HOUR_IN_A_DAY
test_length = test_day_num * HOUR_IN_A_DAY
train_length = data_hour_length - validation_length - test_length

ir_train_data = ir_data[0:train_length]
ir_validation_data = ir_data[train_length:train_length+validation_length]
ir_test_data = ir_data[train_length+validation_length:]

mete_train_data = mete_data[0:train_length]
mete_validation_data = mete_data[train_length:train_length+validation_length]
mete_test_data = mete_data[train_length+validation_length:]

target_train_data = target_data[0:train_length]
target_validation_data = target_data[train_length:train_length+validation_length]
target_test_data = target_data[train_length+validation_length:]

if not os.path.exists('./train/'):
	os.mkdir('./train/')
if not os.path.exists('./validation/'):
	os.mkdir('./validation/')
if not os.path.exists('./test/'):
	os.mkdir('./test/')

#save the data
np.savetxt('./train/ir_train_data.csv', ir_train_data, fmt='%.4f', delimiter=',')
np.savetxt('./validation/ir_validation_data.csv', ir_validation_data, fmt='%.4f', delimiter=',')
np.savetxt('./test/ir_test_data.csv', ir_test_data, fmt='%.4f', delimiter=',')

np.savetxt('./train/mete_train_data.csv', mete_train_data, fmt='%.4f', delimiter=',')
np.savetxt('./validation/mete_validation_data.csv', mete_validation_data, fmt='%.4f', delimiter=',')
np.savetxt('./test/mete_test_data.csv', mete_test_data, fmt='%.4f', delimiter=',')

np.savetxt('./train/target_train_data.csv', target_train_data, fmt='%.4f', delimiter=',')
np.savetxt('./validation/target_validation_data.csv', target_validation_data, fmt='%.4f', delimiter=',')
np.savetxt('./test/target_test_data.csv', target_test_data, fmt='%.4f', delimiter=',')

#generate a README
with open('README', 'w') as fp:
	fp.write("SUMMARY\n")
	fp.write("="*80)
	fp.write("\n\n")

	fp.write("This is an auto generate file by the pre-process file\n")
	fp.write("author: WANG Kejie<wang_kejie@foxmail.com>\n")
	fp.write("Generating time:"+time.strftime('%Y-%m-%d %X', time.localtime())+"\n")
	fp.write("\n\n")

	fp.write("Dataset Info\n")
	fp.write("="*80 + "\n")
	fp.write("The excel file is the source dataset and use the preprocess python script to generate the train, validation and test data\n")
	fp.write("Train set data length: %d days / %d hours\n" %(train_length/HOUR_IN_A_DAY, train_length))
	fp.write("Validation set data length: %d days / %d hours\n" %(validation_length/HOUR_IN_A_DAY, validation_length))
	fp.write("Test set data length: %d days / %d hours\n" %(test_length/HOUR_IN_A_DAY, test_length))
