# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

HOUR_IN_A_DAY = 24
MISSING_VALUE = -99999

ir_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/ir_train_data.csv"
mete_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv"
sky_cam_train_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/train/sky_cam_train_data.csv"
target_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"

ir_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/ir_validation_data.csv"
mete_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/mete_validation_data.csv"
sky_cam_validation_data = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/validation/sky_cam_validation_data.csv"
target_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"

ir_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/ir_test_data.csv"
mete_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv"
sky_cam_test_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/test/sky_cam_test_data.csv"
target_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"

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

    def _get_valid_index(self, ir_features, mete_features, sky_cam_features, targets):
        """
        @brief get the valida index of the features since there are some missing value in some feature and target
        @param ir_features, mete_features, sky_cam_features: the three kinds of feautures
        @return Return a indices indicates the valid features (no missing value) index
        """
        num = len(targets)
        # print num
        missing_index = []
        for i in range(len(targets)):
            if (MISSING_VALUE in ir_features[i]) or \
                (MISSING_VALUE in mete_features[i]) or  \
                (MISSING_VALUE in sky_cam_features[i]) or \
                (MISSING_VALUE in targets[i]):
                missing_index.append(i)
        return np.setdiff1d(np.arange(num), np.array(missing_index))

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

        sky_cam_train_raw_data = np.loadtxt(sky_cam_train_data_path, delimiter=',')
        sky_cam_validation_raw_data = np.loadtxt(sky_cam_validation_data_path, delimiter=',')
        sky_cam_test_raw_data = np.loadtxt(sky_cam_test_data_path, delimiter=',')

        target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
        target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
        target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

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

        self.train_index = self._get_valid_index(self.ir_train_data, self.mete_train_data, self.target_train_data)
        self.validation_index = self._get_valid_index(self.ir_validation_data, self.mete_validation_data, self.target_validation_data)
        self.test_index = self._get_valid_index(self.ir_test_data, self.mete_test_data, self.target_test_data)

        #concatenate all valid data
        ir_raw_valid_data = np.concatenate((ir_train_raw_data[self.train_index], ir_validation_raw_data[self.validation_index], ir_test_raw_data[self.test_index]), axis=0)
        mete_raw_valid_data = np.concatenate((mete_train_raw_data[self.train_index], mete_validation_raw_data[self.validation_index], mete_test_raw_data[self.test_index]), axis=0)
        sky_cam_raw_valid_data = np.concatenate((sky_cam_train_raw_data[self.train_index], sky_cam_validation_raw_data[self.validation_index], sky_cam_test_raw_data[self.test_index]), axis=0)

        #feature scale
        ir_mean = np.mean(ir_raw_valid_data, axis=0)
        ir_std = np.std(ir_raw_valid_data, axis=0)
        self.ir_train_data = (self.ir_train_data - ir_mean) / ir_std
        self.ir_validation_data = (self.ir_validation_data - ir_mean) / ir_std
        self.ir_test_data = (self.ir_test_data - ir_mean) / ir_std

        mete_mean = np.std(mete_raw_valid_data, axis=0)
        mete_std = np.std(mete_raw_valid_data, axis=0)
        self.mete_train_data = (self.mete_train_data - mete_mean) / mete_std
        self.mete_validation_data = (self.ir_vamete_ation_data - mete_mean) / mete_std
        self.mete_test_data = (selfmete__test_data - mete_mean) / mete_std

        sky_cam_mean = np.mean(sky_cam_raw_valid_data, axis=0)
        sky_cam_std = np.std(sky_cam_raw_valid_data, axis=0)
        self.sky_cam_train_data = (self.sky_cam_train_data - sky_cam_mean) / sky_cam_std
        self.sky_cam_validation_data = (self.sky_cam_validation_data - sky_cam_mean) / sky_cam_std
        self.sky_cam_test_data = (self.sky_cam_test_data - sky_cam_mean) / sky_cam_std

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
        print "use", config.n_step, "hours to predict the next ", config.n_target, " consecutive hours"
        print "\n\n"

    def next_batch(self):
        """return a batch of train and target data
        @return ir_data_batch: [batch_size, n_step, n_input]
        @return mete_data_batch:  [batch_size, n_step, n_input]
        @return target_data_batch: [n_model, batch_size, n_target]
        """
        index = np.random.choice(self.train_index, self.batch_size)
        # index = np.random.random_integers(0, self.train_num-1, size=(self.batch_size))
        ir_batch_data = self.ir_train_data[index]
        mete_batch_data = self.mete_train_data[index]
        sky_cam_batch_data = self.sky_cam_batch_data[index]
        target_batch_data = self.target_train_data[index]

        return ir_batch_data, \
                mete_batch_data, \
                sky_cam_batch_data, \
                target_batch_data

    def get_train_set(self):
        return self.ir_train_data[self.train_index], \
                self.mete_train_data[self.train_index], \
                self.sky_cam_train_data[self.train_index]
                self.target_train_data[self.train_index]

    #The returned validataion and test set:
    #ir_data and mete_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    def get_validation_set(self):
        return self.ir_validation_data[0:self.validataion_num], \
                self.mete_validation_data[0:self.validataion_num], \
                self.sky_cam_validation_data[0:self.validataion_num]
                self.target_validation_data[0:self.validataion_num]

    def get_test_set(self, test_num):
        return self.ir_test_data[0:test_num], \
                self.mete_test_data[0:test_num], \
                self.sky_cam_test_data[0:test_num]
                self.target_test_data[0:test_num]
