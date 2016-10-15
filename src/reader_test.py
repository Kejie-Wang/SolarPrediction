import solar_prediction_reader
import solar_prediction_model

config = solar_prediction_model.Config()
reader = solar_prediction_reader.Reader(config.data_path, config)

# for i in range(5):
# 	batch = reader.next_batch()
# 	print batch[2][0]


#get_test function test
for i in range(5):
	print reader.get_test_set()
