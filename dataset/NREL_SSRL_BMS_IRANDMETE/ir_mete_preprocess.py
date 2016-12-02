"""
nrel_solar_preprocess.py

"""
__author__  = "WANG Kejie"
__email__ = "wang_kejie@foxmail.com"

import xlrd
import json
from collections import namedtuple
import numpy as np
import time
import zipfile

HOUR_IN_A_DAY = 24

#load the configuratiuon
fp = open('../../config.json')
config = json.load(fp, object_hook=lambda d:namedtuple('X', d.keys())(*d.values()))
fp.close()

wb = xlrd.open_workbook('NREL_Solar_Dataset.xlsx')
sheet = wb.sheets()[0]

solar_data = list()
temp_data = list()
target_data = list()

for i in range(sheet.ncols):
	col = sheet.col_values(i)
	head = col.pop(0)
	if head in config.input_group_solar:
		solar_data.append(col)
	if head in config.input_group_temp:
		temp_data.append(col)
	if head in config.target_group:
		target_data.append(col)

solar_data = zip(*solar_data)
temp_data = zip(*temp_data)
target_data = zip(*target_data)

data_hour_length = len(solar_data)
data_day_length = data_hour_length / HOUR_IN_A_DAY
train_length = int(data_day_length * config.train_prop) * HOUR_IN_A_DAY
validation_length = int(data_day_length * config.validation_prop) * HOUR_IN_A_DAY
test_length = data_hour_length - train_length - validation_length

solar_train_data = solar_data[0:train_length]
solar_validation_data = solar_data[train_length:train_length+validation_length]
solar_test_data = solar_data[train_length+validation_length:]

temp_train_data = temp_data[0:train_length]
temp_validation_data = temp_data[train_length:train_length+validation_length]
temp_test_data = temp_data[train_length+validation_length:]

target_train_data = target_data[0:train_length]
target_validation_data = target_data[train_length:train_length+validation_length]
target_test_data = target_data[train_length+validation_length:]

np.savetxt('NREL_Solar_Train_Data.csv', solar_train_data, fmt='%.4f', delimiter=',')
np.savetxt('NREL_Solar_Validation_Data.csv', solar_validation_data, fmt='%.4f', delimiter=',')
np.savetxt('NREL_Solar_Test_Data.csv', solar_test_data, fmt='%.4f', delimiter=',')

np.savetxt('NREL_Temp_Train_Data.csv', temp_train_data, fmt='%.4f', delimiter=',')
np.savetxt('NREL_Temp_Validation_Data.csv', temp_validation_data, fmt='%.4f', delimiter=',')
np.savetxt('NREL_Temp_Test_Data.csv', temp_test_data, fmt='%.4f', delimiter=',')

np.savetxt('NREL_Target_Train_Data.csv', target_train_data, fmt='%.4f', delimiter=',')
np.savetxt('NREL_Target_Validation_Data.csv', target_validation_data, fmt='%.4f', delimiter=',')
np.savetxt('NREL_Target_Test_Data.csv', target_test_data, fmt='%.4f', delimiter=',')

with open('README', 'w') as fp:
	fp.write("SUMMARY\n")
	fp.write("="*80)
	fp.write("\n\n")

	fp.write("This is an auto generate file by the pre-preocess file\n")
	fp.write("author: WANG Kejie<wang_kejie@foxmail.com>\n")
	fp.write("Generating time:"+time.strftime('%Y-%m-%d %X', time.localtime())+"\n")
	fp.write("\n\n")

	fp.write("DataSet Info\n")
	fp.write("="*80 + "\n")
	fp.write("The excel file is the source dataset and ues the preprocess python script to generate the train,validation and test data\n")
	fp.write("Train set data length: %d days / %d hours\n" %(train_length/HOUR_IN_A_DAY, train_length))
	fp.write("Validation set data length: %d days / %d hours\n" %(validation_length/HOUR_IN_A_DAY, validation_length))
	fp.write("Test set data length: %d days / %d hours\n" %(test_length/HOUR_IN_A_DAY, test_length))
