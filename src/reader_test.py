import solar_prediction_reader
import solar_prediction_model
import numpy as np
import json
from collections import namedtuple
#get the config
fp = open('../config.json')
config = json.load(fp, object_hook=lambda d:namedtuple('X', d.keys())(*d.values()))
fp.close()
reader = solar_prediction_reader.Reader(config.data_path, config)
	
# for i in range(5):
# 	batch = reader.next_batch()
# 	print batch[2][0]


# pattern = reader.get_pattern(6)
# for i in pattern:
# 	print i


# #get_test function test
# for i in range(1):
# 	test = reader.get_test_set(1)
# 	print test[0]
# 	print "*"*40
# 	print test[2]
