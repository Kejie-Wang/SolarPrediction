# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

HOUR_IN_A_DAY = 24

ir_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/ir_train_data.csv"
mete_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/mete_train_data.csv"
target_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/target_train_data.csv"

ir_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/ir_validation_data.csv"
mete_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/mete_validation_data.csv"
target_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/target_validation_data.csv"

ir_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/ir_test_data.csv"
mete_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/mete_test_data.csv"
target_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/target_test_data.csv"


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
        ir_train_raw_data = np.loadtxt(ir_train_data_path, delimiter=',', ndmin=2)
        ir_validation_raw_data = np.loadtxt(ir_validation_data_path, delimiter=',', ndmin=2)
        ir_test_raw_data = np.loadtxt(ir_test_data_path, delimiter=',', ndmin=2)

        mete_train_raw_data = np.loadtxt(mete_train_data_path, delimiter=',', ndmin=2)
        mete_validation_raw_data = np.loadtxt(mete_validation_data_path, delimiter=',', ndmin=2)
        mete_test_raw_data = np.loadtxt(mete_test_data_path, delimiter=',', ndmin=2)

        target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
        target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
        target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

        ir_raw_data = np.concatenate((ir_train_raw_data, ir_validation_raw_data, ir_test_raw_data), axis=0)
        mete_raw_data = np.concatenate((mete_train_raw_data, mete_validation_raw_data, mete_test_raw_data), axis=0)

        #scale on the all dataset
        # ir_raw_data_mean = np.mean(ir_raw_data, axis=0)
        # mete_raw_data_mean = np.mean(mete_raw_data, axis=0)
        # ir_raw_data_stddev = np.std(ir_raw_data, axis=0)
        # mete_raw_data_stddev = np.std(mete_raw_data, axis=0)
        # ir_train_raw_data = (ir_train_raw_data - ir_raw_data_mean) / ir_raw_data_stddev
        # ir_validation_raw_data = (ir_validation_raw_data - ir_raw_data_mean) / ir_raw_data_stddev
        # ir_test_raw_data = (ir_test_raw_data - ir_raw_data_mean) / ir_raw_data_stddev
        # mete_train_raw_data = (mete_train_raw_data - mete_raw_data_mean) / mete_raw_data_stddev
        # mete_validation_raw_data = (mete_validation_raw_data - mete_raw_data_mean) / mete_raw_data_stddev
        # mete_test_raw_data = (mete_test_raw_data - mete_raw_data_mean) / mete_raw_data_stddev

        #scale on the each hours on a day
        # for i in range(HOUR_IN_A_DAY):
        #     #scale on ir data
        #     index = np.arange(i, len(ir_raw_data), HOUR_IN_A_DAY)
        #     mean = np.mean(ir_raw_data[index], axis=0)
        #     stddev = np.std(ir_raw_data[index], axis=0)
        #
        #     index = np.arange(i, len(ir_train_raw_data), HOUR_IN_A_DAY)
        #     ir_train_raw_data[index] = (ir_train_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(ir_validation_raw_data), HOUR_IN_A_DAY)
        #     ir_validation_raw_data[index] = (ir_validation_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(ir_test_raw_data), HOUR_IN_A_DAY)
        #     ir_test_raw_data[index] = (ir_test_raw_data[index]-mean) / stddev
        #
        #     #scale on mete data
        #     index = np.arange(i, len(mete_raw_data), HOUR_IN_A_DAY)
        #     mean = np.mean(mete_raw_data[index], axis=0)
        #     stddev = np.std(mete_raw_data[index], axis=0)
        #
        #     index = np.arange(i, len(mete_train_raw_data), HOUR_IN_A_DAY)
        #     mete_train_raw_data[index] = (mete_train_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(mete_validation_raw_data), HOUR_IN_A_DAY)
        #     mete_validation_raw_data[index] = (mete_validation_raw_data[index]-mean) / stddev
        #     index = np.arange(i, len(mete_test_raw_data), HOUR_IN_A_DAY)
        #     mete_test_raw_data[index] = (mete_test_raw_data[index]-mean) / stddev

        #feature eshape
        #feature reshape: accumulate several(n_step) features into a new feature for the input the lstm
        #target reshape: align the target with the input feature
        self.ir_train_data = self._feature_reshape(ir_train_raw_data, config.data_step, config.n_step)
        self.ir_validation_data = self._feature_reshape(ir_validation_raw_data, config.data_step, config.n_step)
        self.ir_test_data = self._feature_reshape(ir_test_raw_data, config.data_step, config.n_step)

        self.mete_train_data = self._feature_reshape(mete_train_raw_data, config.data_step, config.n_step)
        self.mete_validation_data = self._feature_reshape(mete_validation_raw_data, config.data_step, config.n_step)
        self.mete_test_data = self._feature_reshape(mete_test_raw_data, config.data_step, config.n_step)

        self.target_train_data = self._target_reshape(target_train_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_validation_data = self._target_reshape(target_validation_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_test_data = self._target_reshape(target_test_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)

        #CAUTIOUS: the length of the ir_tarin_data and target_train_data may be differnet
        #the length of mete_test_data may be more short
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
        @return ir_data_batch: [batch_size, n_step, n_input]
        @return mete_data_batch:  [batch_size, n_step, n_input]
        @return target_data_batch: [n_model, batch_size, n_target]
        """
        index = np.random.random_integers(0, self.train_num-1, size=(self.batch_size))
        ir_batch_data = self.ir_train_data[index]
        mete_batch_data = self.mete_train_data[index]
        target_batch_data = self.target_train_data[index]

        return ir_batch_data, mete_batch_data, target_batch_data

    def get_train_set(self):
        return self.ir_train_data[0:self.train_num], \
                self.mete_train_data[0:self.train_num], \
                self.target_train_data[0:self.train_num]

    #The returned validataion and test set:
    #ir_data and mete_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    def get_validation_set(self):
        return self.ir_validation_data[0:self.validataion_num], \
                self.mete_validation_data[0:self.validataion_num], \
                self.target_validation_data[0:self.validataion_num]

    def get_test_set(self, test_num):
        return self.ir_test_data[0:test_num], \
                self.mete_test_data[0:test_num], \
                self.target_test_data[0:test_num]
