# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

HOUR_IN_A_DAY = 24

class Reader:
    def _feature_scale(self, features):
        features_num = len(features)
        for i in range(HOUR_IN_A_DAY):
            index = np.arange(i, features_num, HOUR_IN_A_DAY)
            mean = np.mean(features[index], axis=0)
            stddev = np.std(features[index], axis=0)
            features[index] = (features[index]-mean) / stddev
        return features

    def _feature_reshape(self, features, data_step, n_step):
        shape_features = []
        for ptr in range(n_step, len(features), data_step):
            shape_features.append(features[ptr - n_step:ptr])
        return np.array(shape_features)

    def _target_reshape(self, targets, data_step, n_step, h_ahead, n_target):
        shape_targets = []
        for ptr in range(n_step + h_ahead + n_target, len(targets), data_step):
            shape_targets.append(targets[ptr - n_target:ptr])
        return np.array(shape_targets)

    def __init__(self, config):
        """
        """

        #load data
        solar_train_raw_data = np.loadtxt(config.solar_train_data_path, delimiter=',', ndmin=2)
        solar_validation_raw_data = np.loadtxt(config.solar_validation_data_path, delimiter=',', ndmin=2)
        solar_test_raw_data = np.loadtxt(config.solar_test_data_path, delimiter=',', ndmin=2)

        temp_train_raw_data = np.loadtxt(config.temp_train_data_path, delimiter=',', ndmin=2)
        temp_validation_raw_data = np.loadtxt(config.temp_validation_data_path, delimiter=',', ndmin=2)
        temp_test_raw_data = np.loadtxt(config.temp_test_data_path, delimiter=',', ndmin=2)

        target_train_raw_data = np.loadtxt(config.target_train_data_path, delimiter=',')
        target_validation_raw_data = np.loadtxt(config.target_validation_data_path, delimiter=',')
        target_test_raw_data = np.loadtxt(config.target_test_data_path, delimiter=',')

        solar_raw_data = np.concatenate((solar_train_raw_data, solar_validation_raw_data, solar_test_raw_data), axis=0)
        temp_raw_data = np.concatenate((temp_train_raw_data, temp_validation_raw_data, temp_test_raw_data), axis=0)

        #scale on the all dataset
        # solar_raw_data_mean = np.mean(solar_raw_data, axis=0)
        # temp_raw_data_mean = np.mean(temp_raw_data, axis=0)
        # solar_raw_data_stddev = np.std(solar_raw_data, axis=0)
        # temp_raw_data_stddev = np.std(temp_raw_data, axis=0)
        # solar_train_raw_data = (solar_train_raw_data - solar_raw_data_mean) / solar_raw_data_stddev
        # solar_validation_raw_data = (solar_validation_raw_data - solar_raw_data_mean) / solar_raw_data_stddev
        # solar_test_raw_data = (solar_test_raw_data - solar_raw_data_mean) / solar_raw_data_stddev
        # temp_train_raw_data = (temp_train_raw_data - temp_raw_data_mean) / temp_raw_data_stddev
        # temp_validation_raw_data = (temp_validation_raw_data - temp_raw_data_mean) / temp_raw_data_stddev
        # temp_test_raw_data = (temp_test_raw_data - temp_raw_data_mean) / temp_raw_data_stddev

        #scale on the each hours on a day
        # for i in range(HOUR_IN_A_DAY):
        #     #scale on solar data
        #     index = np.arange(i, len(solar_raw_data), HOUR_IN_A_DAY)
        #     mean = np.mean(solar_raw_data[index], axis=0)
        #     stddev = np.std(solar_raw_data[index], axis=0)
        #
        #     index = np.arange(i, len(solar_train_raw_data), HOUR_IN_A_DAY)
        #     solar_train_raw_data[index] = (solar_train_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(solar_validation_raw_data), HOUR_IN_A_DAY)
        #     solar_validation_raw_data[index] = (solar_validation_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(solar_test_raw_data), HOUR_IN_A_DAY)
        #     solar_test_raw_data[index] = (solar_test_raw_data[index]-mean) / stddev
        #
        #     #scale on temp data
        #     index = np.arange(i, len(temp_raw_data), HOUR_IN_A_DAY)
        #     mean = np.mean(temp_raw_data[index], axis=0)
        #     stddev = np.std(temp_raw_data[index], axis=0)
        #
        #     index = np.arange(i, len(temp_train_raw_data), HOUR_IN_A_DAY)
        #     temp_train_raw_data[index] = (temp_train_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(temp_validation_raw_data), HOUR_IN_A_DAY)
        #     temp_validation_raw_data[index] = (temp_validation_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(temp_test_raw_data), HOUR_IN_A_DAY)
        #     temp_test_raw_data[index] = (temp_test_raw_data[index]-mean) / stddev

        #feature eshape
        #feature reshape: accumulate several(n_step) features into a new feature for the input the lstm
        #target reshape: align the target with the input feature
        self.solar_train_data = self._feature_reshape(solar_train_raw_data, config.data_step, config.n_step)
        self.solar_validation_data = self._feature_reshape(solar_validation_raw_data, config.data_step, config.n_step)
        self.solar_test_data = self._feature_reshape(solar_test_raw_data, config.data_step, config.n_step)

        self.temp_train_data = self._feature_reshape(temp_train_raw_data, config.data_step, config.n_step)
        self.temp_validation_data = self._feature_reshape(temp_validation_raw_data, config.data_step, config.n_step)
        self.temp_test_data = self._feature_reshape(temp_test_raw_data, config.data_step, config.n_step)

        self.target_train_data = self._target_reshape(target_train_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_validation_data = self._target_reshape(target_validation_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_test_data = self._target_reshape(target_test_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)

        #CAUTIOUS: the length of the solar_tarin_data and target_train_data may be differnet
        #the length of temp_test_data may be more short
        #and thus we must use the target data to compute the number
        self.train_num = len(self.target_train_data)
        self.validataion_num = len(self.target_validation_data)
        self.test_num = len(self.target_test_data)

        self.batch_size = config.batch_size

        # self.index = np.random.random_integers(0, self.train_num-1, size=(self.batch_size))
        #print the dataset info
        print "Dataset info"
        print "="*80
        print "train number:", self.train_num
        print "validation number:", self.validataion_num
        print "test number", self.test_num
        print "batch size:", self.batch_size
        print "use ", config.n_step, "hours to predict the next ", config.n_target, " consecutive hours"
        print "\n\n"

    def next_batch(self):
        """return a batch of train and target data
        @return solar_data_batch: [batch_size, n_step, n_input]
        @return temp_data_batch:  [batch_size, n_step, n_input]
        @return target_data_batch: [n_model, batch_size, n_target]
        """
        index = np.random.random_integers(0, self.train_num-1, size=(self.batch_size))
        solar_batch_data = self.solar_train_data[index]
        temp_batch_data = self.temp_train_data[index]
        target_batch_data = self.target_train_data[index]

        return solar_batch_data, temp_batch_data, target_batch_data

    def get_train_set(self):
        return self.solar_train_data[0:self.train_num], \
                self.temp_train_data[0:self.train_num], \
                self.target_train_data[0:self.train_num]

    #The returned validataion and test set:
    #solar_data and temp_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    def get_validation_set(self):
        return self.solar_validation_data[0:self.validataion_num], \
                self.temp_validation_data[0:self.validataion_num], \
                self.target_validation_data[0:self.validataion_num]

    def get_test_set(self, test_num):
        return self.solar_test_data[0:test_num], \
                self.temp_test_data[0:test_num], \
                self.target_test_data[0:test_num]
