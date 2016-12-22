# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import cv2

HOUR_IN_A_DAY = 24
MISSING_VALUE = -99999

ir_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/ir_train_data.csv"
mete_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv"
sky_cam_train_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/train/sky_cam_train_data.csv"
target_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"

ir_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/ir_validation_data.csv"
mete_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/mete_validation_data.csv"
sky_cam_validation_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/validation/sky_cam_validation_data.csv"
target_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"

ir_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/ir_test_data.csv"
mete_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv"
sky_cam_test_data_path = "../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/test/sky_cam_test_data.csv"
target_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"

sky_cam_raw_data_path = '../dataset/NREL_SSRL_BMS_SKY_CAM/raw_data/SSRL_SKY_CAM_IMAGE/'

class Reader:

    def _feature_reshape(self, features, data_step, n_step):
        """
        @brief aggregate the multiple features (a successive time features) as a new feature to predict the following time target
        @param features The input features in shape [features_num, feature_dim], each of which is a feature of a specific time
        @param data_step The interval between two new features
        @param n_step The length of the time in the lstm
        @return a new batch shaped features in shape [new_feature_num, n_step, feature_dim]
        """
        shape_features = []
        for ptr in range(n_step, len(features), data_step):
            shape_features.append(features[ptr - n_step:ptr])
        return np.array(shape_features)

    def _target_reshape(self, targets, data_step, n_step, h_ahead, n_target):
        """
        @brief aggregate the multiple features as a new target for multiple output and synchronize the target with the features
        @param targets The input targets in shape [target_num, target_dim] ##now target_num=1
        @param n_step  The interval between two new features
        @param h_ahead The interval of the feature and target
        @param n_target The  number of target for a prediction
        @return an aggregated and synchronized targets in shape [new_target_num, n_target, target_dim]
        """
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
                (True in np.isnan(sky_cam_features[i])) or \
                (MISSING_VALUE in targets[i]):
                missing_index.append(i)
        print missing_index
        return np.setdiff1d(np.arange(num), np.array(missing_index))

    def __init__(self, config):
        """
        The constructor of the class Reader
        load the train, validation and test data from the dataset and call the function to aggregate and synchronize the features and target
        filter the some points with the missing value
        and do the feature pre-processing (now just scale the feature into mean is 0 and stddev is 1.0)
        """

        #load data
        ir_train_raw_data = np.loadtxt(ir_train_data_path, delimiter=',', ndmin=2)
        ir_validation_raw_data = np.loadtxt(ir_validation_data_path, delimiter=',', ndmin=2)
        ir_test_raw_data = np.loadtxt(ir_test_data_path, delimiter=',', ndmin=2)

        mete_train_raw_data = np.loadtxt(mete_train_data_path, delimiter=',', ndmin=2)
        mete_validation_raw_data = np.loadtxt(mete_validation_data_path, delimiter=',', ndmin=2)
        mete_test_raw_data = np.loadtxt(mete_test_data_path, delimiter=',', ndmin=2)

        sky_cam_train_raw_data = np.array(np.loadtxt(sky_cam_train_data_path, delimiter=',', dtype='float'), dtype='int')
        sky_cam_validation_raw_data = np.array(np.loadtxt(sky_cam_validation_data_path, delimiter=',', dtype='float'), dtype='int')
        sky_cam_test_raw_data = np.array(np.loadtxt(sky_cam_test_data_path, delimiter=',', dtype='float'), dtype='int')

        target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
        target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
        target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

        #feature reshape
        #feature reshape: accumulate several(n_step) features into a new feature for the input the lstm
        #target reshape: align the target with the input feature
        self.ir_train_data = self._feature_reshape(ir_train_raw_data, config.data_step, config.n_step)
        self.ir_validation_data = self._feature_reshape(ir_validation_raw_data, config.data_step, config.n_step)
        self.ir_test_data = self._feature_reshape(ir_test_raw_data, config.data_step, config.n_step)

        self.mete_train_data = self._feature_reshape(mete_train_raw_data, config.data_step, config.n_step)
        self.mete_validation_data = self._feature_reshape(mete_validation_raw_data, config.data_step, config.n_step)
        self.mete_test_data = self._feature_reshape(mete_test_raw_data, config.data_step, config.n_step)

        self.sky_cam_train_data = self._feature_reshape(sky_cam_train_raw_data, config.data_step, config.n_step)
        self.sky_cam_validation_data = self._feature_reshape(sky_cam_validation_raw_data, config.data_step, config.n_step)
        self.sky_cam_test_data = self._feature_reshape(sky_cam_test_raw_data, config.data_step, config.n_step)

        self.target_train_data = self._target_reshape(target_train_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_validation_data = self._target_reshape(target_validation_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        self.target_test_data = self._target_reshape(target_test_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)

        self.train_index = self._get_valid_index(self.ir_train_data, self.mete_train_data, self.sky_cam_train_data, self.target_train_data)
        self.validation_index = self._get_valid_index(self.ir_validation_data, self.mete_validation_data, self.sky_cam_validation_data, self.target_validation_data)
        self.test_index = self._get_valid_index(self.ir_test_data, self.mete_test_data, self.sky_cam_test_data, self.target_test_data)

        #concatenate all valid data
        ir_raw_valid_data = np.concatenate((ir_train_raw_data[self.train_index], ir_validation_raw_data[self.validation_index], ir_test_raw_data[self.test_index]), axis=0)
        mete_raw_valid_data = np.concatenate((mete_train_raw_data[self.train_index], mete_validation_raw_data[self.validation_index], mete_test_raw_data[self.test_index]), axis=0)

        #feature scale
        ir_mean = np.mean(ir_raw_valid_data, axis=0)
        ir_std = np.std(ir_raw_valid_data, axis=0)
        self.ir_train_data = (self.ir_train_data - ir_mean) / ir_std
        self.ir_validation_data = (self.ir_validation_data - ir_mean) / ir_std
        self.ir_test_data = (self.ir_test_data - ir_mean) / ir_std

        mete_mean = np.std(mete_raw_valid_data, axis=0)
        mete_std = np.std(mete_raw_valid_data, axis=0)
        self.mete_train_data = (self.mete_train_data - mete_mean) / mete_std
        self.mete_validation_data = (self.mete_validation_data - mete_mean) / mete_std
        self.mete_test_data = (self.mete_test_data - mete_mean) / mete_std

        #CAUTIOUS: the length of the ir_tarin_data and target_train_data may be differnet
        #the length of mete_test_data may be more short
        #and thus we must use the target data to compute the number
        self.train_num = len(self.target_train_data)
        self.validataion_num = len(self.target_validation_data)
        self.test_num = len(self.target_test_data)

        self.batch_size = config.batch_size

        self.n_step = config.n_step
        self.width = config.width
        self.height = config.height

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

    def path2image(self, data, index):
        img_list = []
        for idx in index:
            img = []
            for i in range(self.n_step):
                if data[idx, i] == -11111:
                    img.append(np.zeros((self.height,self.width)))
                else:
                    #print '3'
                    filename = str(int(data[idx, i]))
                    #print 'lllllllllllllll:', data[idx, i]
                    y = filename[:4]
                    m = filename[4:6]
                    d = filename[6:8]
                    #h = filename[8:]
                    #print filename, y, m, d, h
                    path = sky_cam_raw_data_path + str(y) + '/' + str(m) + '/' + str(d) + '/' + str(filename) + '.jpg'
                    #/ home / lcc / code / python / SolarPrediction / dataset / NREL_SSRL_BMS_SKY_CAM / SSRL_SKY / 2008 / 01 / 01
                    tmp = cv2.resize(cv2.imread(path, 0), (self.height, self.width))
                    #print 'TMMMMMMMMMMMMMP:', tmp.shape
                    img.append(tmp)
            img_list.append(img)
        return np.array(img_list)


    def next_batch(self):
        """
        @brief return a batch of train and target data
        @return ir_data_batch: [batch_size, n_step, n_input]
        @return mete_data_batch:  [batch_size, n_step, n_input]
        @return target_data_batch: [n_model, batch_size, n_target]
        """
        index = np.random.choice(self.train_index, self.batch_size, replace=False)
        # index = np.random.random_integers(0, self.train_num-1, size=(self.batch_size))
        ir_batch_data = self.ir_train_data[index]
        mete_batch_data = self.mete_train_data[index]
        sky_cam_batch_data = self.path2image(self.sky_cam_train_data, index)
        target_batch_data = self.target_train_data[index]

        return ir_batch_data, \
                mete_batch_data, \
                sky_cam_batch_data, \
                target_batch_data

    def get_train_set(self):
        """
        @brief return the total dataset
        """
        return self.ir_train_data[self.train_index], \
                self.mete_train_data[self.train_index], \
                self.path2image(self.sky_cam_train_data, self.train_index), \
                self.target_train_data[self.train_index]

    #The returned validataion and test set:
    #ir_data and mete_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    def get_validation_set(self):
        """
        @brief return the total validation dataset
        """
        return self.ir_validation_data[self.validation_index], \
                self.mete_validation_data[self.validation_index], \
                self.path2image(self.sky_cam_validation_data, self.validation_index), \
                self.target_validation_data[self.validation_index]

    def get_test_set(self):
        """
        @brief return a test set in the specific test num
        @param test_num The number of test set to return
        """
        return self.ir_test_data[self.test_index], \
                self.mete_test_data[self.test_index], \
                self.path2image(self.sky_cam_test_data, self.test_index), \
                self.target_test_data[self.test_index]
