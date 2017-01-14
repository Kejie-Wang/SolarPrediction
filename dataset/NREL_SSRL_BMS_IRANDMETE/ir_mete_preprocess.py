from __future__ import division

# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '20/11/2016'

"""
ir_mete_preprocess.py
"""
__author__  = "WANG Kejie"
__email__ = "wang_kejie@foxmail.com"

import numpy as np
import zipfile
from io import StringIO
import os

MINUTE_IN_A_HOUR = 60
HOUR_IN_A_DAY = 24
MONTH_IN_A_YEAR = 12
MISSING_VALUE = -99999

#dataset range configuration
year_start = 2000
month_start = 11
year_end = 2016
month_end = 11

#data path
raw_data_path = "./raw_data/"
save_data_path = "./input_data/" + str(year_start) + str(month_start).rjust(2, '0') + str(year_end) + str(month_end).rjust(2, '0') + '/'

#field id path params
ir_field_file_path = "field/irradiance_common_id.csv"
mete_field_file_path = "field/meteorological_common_id.csv"
target_field_file_path = "field/target_id.csv"

#load the common field of the irradiance and meteorological
ir_field = np.loadtxt(ir_field_file_path, dtype='str', delimiter=',', comments=None, ndmin=2)[:,1].astype(int)
mete_field = np.loadtxt(mete_field_file_path, dtype='str', delimiter=',', comments=None, ndmin=2)[:,1].astype(int)
target_field = np.loadtxt(target_field_file_path, dtype='str', delimiter=',', comments=None, ndmin=2)[:,1].astype(int)
#index starts from zero
ir_field = ir_field - 1
mete_field = mete_field - 1
target_field = target_field - 1

ir_data = list()
mete_data = list()
target_data = list()

yit, mit = year_start, month_start
while (yit < year_end) or (yit==year_end and mit<=month_end):
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
	data = np.reshape(data, (-1, MINUTE_IN_A_HOUR, data.shape[1]))
	missing_value_num = np.sum((data != MISSING_VALUE).astype(np.float32), axis=1)
	data[data == MISSING_VALUE] = 0
	data = np.sum(data, axis=1) / missing_value_num
	data[data == np.inf] = MISSING_VALUE
	# data = np.mean(np.reshape(data, (-1, MINUTE_IN_A_HOUR, data.shape[1])), axis=1)
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

if not os.path.exists(save_data_path):
	os.mkdir(save_data_path)
np.savetxt(save_data_path + 'ir_data.csv', ir_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'mete_data.csv', mete_data, fmt='%.4f', delimiter=',')
np.savetxt(save_data_path + 'target_data.csv', target_data, fmt='%.4f', delimiter=',')
