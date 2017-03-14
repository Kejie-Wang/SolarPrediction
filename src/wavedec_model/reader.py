# -*- coding: utf-8 -*-
from __future__ import division
import sys
# append the upper dir into the system path
sys.path.append('../')
import numpy as np
from feature_reader import Feature_Reader
from target_reader import Target_Reader
from similar_day_fea_reader import Similar_Day_Reader
import cv2
import pywt

MINUTES_TO_AVG = 15
HOUR_IN_A_DAY = 24

ir_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/train/ir_train_data.csv"
mete_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/train/mete_train_data.csv"
target_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/train/target_train_hourly_data.csv"
time_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/train/time_train_data.csv"

ir_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/validation/ir_validation_data.csv"
mete_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/validation/mete_validation_data.csv"
target_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/validation/target_validation_hourly_data.csv"
time_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/validation/time_validation_data.csv"

ir_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/test/ir_test_data.csv"
mete_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/test/mete_test_data.csv"
target_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/test/target_test_hourly_data.csv"
time_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/wavedec_input_data/test/time_test_data.csv"

similar_day_train_data_path = '../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/similar_day_train_data.csv'
similar_day_validation_data_path = '../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/similar_day_validation_data.csv'
similar_day_test_data_path = '../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/similar_day_test_data.csv'

hour_index_range = [5, 18]

class Reader:

    def _get_hour_index_and_filter_data(self, max_shift, h_ahead, data_step, data_index):
        # get hour index
        start_hour_index = (max_shift + h_ahead) % HOUR_IN_A_DAY
        hour_index = np.arange(start_hour_index, max(data_index)*data_step+start_hour_index+1, data_step)
        hour_index = hour_index[data_index] % HOUR_IN_A_DAY
        # filter the data
        index = np.logical_and(hour_index>=hour_index_range[0], hour_index<=hour_index_range[1])

        return data_index[index], np.reshape(hour_index[index], [-1, 1])

    def _wavelet_dec(self, data, wavelet, level):
        '''
        data: [batch_size, n_step, feature_num]
        '''
        # transpose the data into [batch_size, feature_num, n_step]
        data = np.transpose(data, [0, 2, 1])
        # [batch_size, feature_num, n_step] ====> a list of coeff has length level+1, each of which is in shape [batch_size, feature_num, n_step_new]
        coeffs = pywt.wavedec(data, wavelet=wavelet, level=level)
        for i in range(len(coeffs)):
            # [batch_size, feature_num, n_step_new] ===> [batch_size, n_step_new, feature_num]
            coeffs[i] = np.transpose(coeffs[i], [0, 2, 1])
        return coeffs


    def __init__(self, config):
        """
        The constructor of the class Reader
        load the train, validation and test data from the dataset and call the function to aggregate and synchronize the features and target
        filter the some points with the missing value
        and do the feature pre-processing (now just scale the feature into mean is 0 and stddev is 1.0)
        """
        data_step = config.data_step
        h_ahead = config.h_ahead
        n_target = config.n_target
        self.modality = config.modality

        n_step_1 = config.n_step_1
        n_step_2 = config.n_step_2

        wavelet = config.wavelet
        level = config.level

        shift = []
        if self.modality[0] == 1:
            shift.append(config.n_shift_1)
        if self.modality[1] == 1:
            shift.append(config.n_shift_2)
        if len(shift) > 0:
            max_shift = max(shift)
        MINUTE_TO_AVG_IN_HOUR = 60 // MINUTES_TO_AVG

        train_index = []; validation_index = []; test_index = []

        # construct the reader
        # construct the first modality reader
        if self.modality[0] == 1:
            shift_day_1 = (max_shift - config.n_shift_1) // data_step; assert (max_shift - config.n_shift_1)%data_step == 0
            ir_feature_reader = Feature_Reader(ir_train_data_path, ir_validation_data_path, ir_test_data_path, \
                n_step_1*MINUTE_TO_AVG_IN_HOUR , data_step*MINUTE_TO_AVG_IN_HOUR, shift_day_1*MINUTE_TO_AVG_IN_HOUR)
            ir_train_index, ir_validation_index, ir_test_index = ir_feature_reader.get_index()
            train_index.append(ir_train_index); validation_index.append(ir_validation_index); test_index.append(ir_test_index)

        # construct the second modality reader
        if self.modality[1] == 1:
            shift_day_2 = (max_shift - config.n_shift_2) // data_step; assert (max_shift - config.n_shift_2)%data_step == 0
            mete_feature_reader = Feature_Reader(mete_train_data_path, mete_validation_data_path, mete_test_data_path, \
                n_step_2*MINUTE_TO_AVG_IN_HOUR, data_step*MINUTE_TO_AVG_IN_HOUR, shift_day_2*MINUTE_TO_AVG_IN_HOUR)
            mete_train_index, mete_validation_index, mete_test_index = mete_feature_reader.get_index()
            train_index.append(mete_train_index); validation_index.append(mete_validation_index); test_index.append(mete_test_index)

        if self.modality[2] == 1:
            # construct the similar day feature reader
            sim_day_reader = Similar_Day_Reader(similar_day_train_data_path, similar_day_validation_data_path, similar_day_test_data_path, \
                                                time_train_data_path, time_validation_data_path, time_test_data_path, target_train_data_path)

        # construct the target reader
        target_reader = Target_Reader(target_train_data_path, target_validation_data_path, target_test_data_path, max_shift, h_ahead, data_step, n_target)
        target_train_index, target_validation_index, target_test_index = target_reader.get_index()
        train_index.append(target_train_index); validation_index.append(target_validation_index); test_index.append(target_test_index)

        # construct the time index read
        time_reader = Target_Reader(time_train_data_path, time_validation_data_path, time_test_data_path, max_shift, h_ahead, data_step, n_target, dtype=np.int)

        # reduce the multiple modality valid index
        train_index = reduce(np.intersect1d, train_index)
        validation_index = reduce(np.intersect1d, validation_index)
        test_index= reduce(np.intersect1d, test_index)

        # get the hour index
        # filter the hour index less than 5 or bigger than 18
        # the irradiance in this range is nearly zero and it need not forecast
        train_index, self.train_hour_index = self._get_hour_index_and_filter_data(max_shift, h_ahead, data_step, train_index)
        validation_index, self.validation_hour_index = self._get_hour_index_and_filter_data(max_shift, h_ahead, data_step, validation_index)
        test_index, self.test_hour_index = self._get_hour_index_and_filter_data(max_shift, h_ahead, data_step, test_index)

        self.n_step_level = []

        # read the first modality data
        if self.modality[0] == 1:
            ir_train_data, ir_validation_data, ir_test_data = ir_feature_reader.get_data(train_index, validation_index, test_index)
            ir_mean, ir_std = ir_feature_reader.get_mean_std()
            self.ir_train_data = self._wavelet_dec((ir_train_data - ir_mean) / ir_std, wavelet, level)
            self.ir_validation_data = self._wavelet_dec((ir_validation_data - ir_mean) / ir_std, wavelet, level)
            self.ir_test_data = self._wavelet_dec((ir_test_data - ir_mean) / ir_std, wavelet, level)
            for d in self.ir_test_data:
                self.n_step_level.append(d.shape[1])

        # read the second modality data
        if self.modality[1] == 1:
            mete_train_data, mete_validation_data, mete_test_data = mete_feature_reader.get_data(train_index, validation_index, test_index)
            mete_mean, mete_std = mete_feature_reader.get_mean_std()
            self.mete_train_data = self._wavelet_dec((mete_train_data - mete_mean) / mete_std, wavelet, level)
            self.mete_validation_data = self._wavelet_dec((mete_validation_data - mete_mean) / mete_std, wavelet, level)
            self.mete_test_data = self._wavelet_dec((mete_test_data - mete_mean) / mete_std, wavelet, level)
            if self.modality[0] == 0:
                for d in self.mete_test_data:
                    self.n_step_level.append(d.shape[1])

        # read the target data
        self.target_train_data, self.target_validation_data, self.target_test_data = target_reader.get_data(train_index, validation_index, test_index)

        self.time_train_data, self.time_validation_data, self.time_test_data = time_reader.get_data(train_index, validation_index, test_index)
        self.time_train_data = self.time_train_data[:,0,:]
        self.time_validation_data = self.time_validation_data[:,0,:]
        self.time_test_data = self.time_test_data[:,0,:]

        if self.modality[2] == 1:
            self.similar_day_train_data, self.similar_day_validation_data, self.similar_day_test_data = \
                sim_day_reader.get_data(self.time_train_data, self.time_validation_data, self.time_test_data)

        #CAUTIOUS: the length of the ir_train_data and target_train_data may be differnet
        #the length of mete_test_data may be more short
        #and thus we must use the target data to compute the number
        self.train_num = len(self.target_train_data)
        self.validation_num = len(self.target_validation_data)
        self.test_num = len(self.target_test_data)

        self.batch_size = config.batch_size

        # self.index = np.random.random_integers(0, self.train_num-1, size=(self.batch_size))
        #print the dataset info
        print '\033[1;31;40m'
        print "Dataset info"
        print "="*80
        print "train number:", self.train_num
        print "validation number:", self.validation_num
        print "test number", self.test_num
        print "batch size:", self.batch_size
        print '\033[0m'

    def get_n_step_level(self):
        return self.n_step_level

    def _get_batch_data(self, data, index):
        batch_data = []
        for i in range(len(data)):  # for each level
            batch_data.append(data[i][index])
        return batch_data

    def next_batch(self):
        """
        @brief return a batch of train and target data
        @return ir_data_batch: [batch_size, n_step, n_input]
        @return mete_data_batch:  [batch_size, n_step, n_input]
        @return target_data_batch: [n_model, batch_size, n_target]
        """
        index = np.random.choice(np.arange(self.train_num), self.batch_size, replace=False)
        batch = []
        if self.modality[0] == 1:
            ir_batch_data = self._get_batch_data(self.ir_train_data, index)
            batch.append(ir_batch_data)
        if self.modality[1] == 1:
            mete_batch_data = self._get_batch_data(self.mete_train_data, index)
            batch.append(mete_batch_data)
        if self.modality[2] == 1:
            batch.append(self.similar_day_train_data[index])
        batch.append(self.train_hour_index[index])
        batch.append(self.target_train_data[index])

        return batch

    def get_train_set(self, start=None, length=None):
        """
        @brief return the total dataset
        """
        if start is None:
            start = 0
        if length is None or start + length > self.train_num:
            length = self.train_num - start
        if length <= 0:
            return []

        index = np.arange(start, start+length)
        train_set = []
        if self.modality[0] == 1:
            ir_batch_data = self._get_batch_data(self.ir_train_data, index)
            train_set.append(ir_batch_data)
        if self.modality[1] == 1:
            mete_batch_data = self._get_batch_data(self.mete_train_data, index)
            train_set.append(mete_batch_data)
        if self.modality[2] == 1:
            train_set.append(self.similar_day_train_data[index])
        train_set.append(self.train_hour_index[index])
        train_set.append(self.target_train_data[index])

        return train_set

    #The returned validataion and test set:
    #ir_data and mete_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    def get_validation_set(self, start=None, length=None):
        """
        @brief return the total validation dataset
        """
        if start is None:
            start = 0
        if length is None or start + length > self.validation_num:
            length = self.validation_num - start
        if length <= 0:
            return []

        index = np.arange(start, start+length)
        validation_set = []
        if self.modality[0] == 1:
            ir_batch_data = self._get_batch_data(self.ir_validation_data, index)
            validation_set.append(ir_batch_data)
        if self.modality[1] == 1:
            mete_batch_data = self._get_batch_data(self.mete_validation_data, index)
            validation_set.append(mete_batch_data)
        if self.modality[2] == 1:
            validation_set.append(self.similar_day_validation_data[index])
        validation_set.append(self.validation_hour_index[index])
        validation_set.append(self.target_validation_data[index])

        return validation_set

    def get_test_set(self, start=None, length=None):
        """
        @brief return a test set in the specific test num
        """
        if start is None:
            start = 0
        if length is None or start + length > self.test_num:
            length = self.test_num - start
        if length <= 0:
            return []

        index = np.arange(start, start+length)
        test_set = []
        if self.modality[0] == 1:
            ir_batch_data = self._get_batch_data(self.ir_test_data, index)
            test_set.append(ir_batch_data)
        if self.modality[1] == 1:
            mete_batch_data = self._get_batch_data(self.mete_test_data, index)
            test_set.append(mete_batch_data)
        if self.modality[2] == 1:
            test_set.append(self.similar_day_test_data[index])
        test_set.append(self.test_hour_index[index])
        test_set.append(self.target_test_data[index])

        return test_set
