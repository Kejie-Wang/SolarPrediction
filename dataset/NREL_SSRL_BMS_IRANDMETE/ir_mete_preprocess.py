from __future__ import division
"""
ir_mete_preprocess.py

"""
__author__  = "WANG Kejie"
__email__ = "wang_kejie@foxmail.com"

import json
from collections import namedtuple
import numpy as np
import time
import zipfile
from io import StringIO
import os

MINUTE_IN_A_HOUR = 60
HOUR_IN_A_DAY = 24
MONTH_IN_A_YEAR = 12

#data path
raw_data_path = "./raw_data/"
save_data_path = "./input_data/"

#field id path params
ir_field_file_path = "field/irradiance_common_id.csv"
mete_field_file_path = "field/meteorological_common_id.csv"
target_field_file_path = "field/target_id.csv"

#load the configuratiuon
fp = open('../../config.json')
config = json.load(fp, object_hook=lambda d:namedtuple('X', d.keys())(*d.values()))
fp.close()

#load the common field of the irradiance and meteorological
ir_field = np.loadtxt(ir_field_file_path, dtype='str', delimiter=',', comments=None, ndmin=2)[:,1].astype(int)
mete_field = np.loadtxt(mete_field_file_path, dtype='str', delimiter=',', comments=None, ndmin=2)[:,1].astype(int)
target_field = np.loadtxt(target_field_file_path, dtype='str', delimiter=',', comments=None, ndmin=2)[:,1].astype(int)
ir_field = ir_field - 1
mete_field = mete_field - 1
target_field = target_field - 1

ir_data = list()
mete_data = list()
target_data = list()

yit, mit = config.year_start, config.month_start
while (yit < config.year_end) or (yit==config.year_end and mit<=config.month_end):
	print yit,mit
	zip_file_path = raw_data_path + str(yit).rjust(4, '0') + "/" + str(yit).rjust(4, '0') + str(mit).rjust(2, '0') + ".zip"
	zf = zipfile.ZipFile(zip_file_path, 'r')
	str_data = zf.read(zf.namelist()[0]).replace('\r\n', '\n')
	data = np.genfromtxt(StringIO(unicode(str_data)), delimiter=',')

	#only select the integer data as the hourly data
	# index = np.arange(59, data.shape[0], MINUTE_IN_A_HOUR)
	# ir_data.append(data[index,:][:,ir_field])
	# mete_data.append(data[index,:][:,mete_field])
	# target_data.append(data[index,:][:,target_field])

	#average the data in a hour
	data = np.mean(np.reshape(data, (-1, MINUTE_IN_A_HOUR, data.shape[1])), axis=1)
	ir_data.append(data[:,ir_field])
	mete_data.append(data[:,mete_field])
	target_data.append(data[:,target_field])

	#next month
	yit = yit + mit // MONTH_IN_A_YEAR
	mit = mit%MONTH_IN_A_YEAR + 1

#concatenate all data
ir_data = np.concatenate(ir_data, axis=0)
mete_data = np.concatenate(mete_data, axis=0)
target_data = np.concatenate(target_data, axis=0)

#
data_hour_length = len(ir_data)
data_day_length = data_hour_length / HOUR_IN_A_DAY
train_length = int(data_day_length * config.train_prop) * HOUR_IN_A_DAY
validation_length = int(data_day_length * config.validation_prop) * HOUR_IN_A_DAY
test_length = data_hour_length - train_length - validation_length

ir_train_data = ir_data[0:train_length]
ir_validation_data = ir_data[train_length:train_length+validation_length]
ir_test_data = ir_data[train_length+validation_length:]

mete_train_data = mete_data[0:train_length]
mete_validation_data = mete_data[train_length:train_length+validation_length]
mete_test_data = mete_data[train_length+validation_length:]

target_train_data = target_data[0:train_length]
target_validation_data = target_data[train_length:train_length+validation_length]
target_test_data = target_data[train_length+validation_length:]

if not os.path.exists(save_data_path):
	os.mkdir(save_data_path)
np.savetxt(save_data_path + 'ir_train_data.csv', ir_train_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'ir_validation_data.csv', ir_validation_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'ir_test_data.csv', ir_test_data, fmt='%.4f', delimiter=',')

np.savetxt(save_data_path + 'mete_train_data.csv', mete_train_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'mete_validation_data.csv', mete_validation_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'mete_test_data.csv', mete_test_data, fmt='%.4f', delimiter=',')

np.savetxt(save_data_path + 'target_train_data.csv', target_train_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'target_validation_data.csv', target_validation_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'target_test_data.csv', target_test_data, fmt='%.4f', delimiter=',')

with open(save_data_path + 'README', 'w') as fp:
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
