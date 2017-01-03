# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

HOUR_IN_A_DAY = 24
MISSING_VALUE = -99999

ir_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/ir_train_data.csv"
mete_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv"
target_train_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"

ir_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/ir_validation_data.csv"
mete_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/mete_validation_data.csv"
target_validation_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"

ir_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/ir_test_data.csv"
mete_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv"
target_test_data_path = "../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"

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

    def _get_valid_index(self, ir_features, mete_features, targets):
        """
        @brief get the valida index of the features since there are some missing value in some feature and target
        @param ir_features, mete_features, sky_cam_features: the three kinds of feautures
        @return Return a indices indicates the valid features (no missing value) index
        """
        num = len(targets)
        missing_index = []
        for i in range(len(targets)):
            if (True in np.isnan(ir_features[i])) or \
                (True in np.isnan(mete_features[i])) or \
                (True in np.isnan(targets[i])):
                missing_index.append(i)
        print missing_index
        return np.setdiff1d(np.arange(num), np.array(missing_index))

    def _add_noise_to_data(self, data, expand_rate, stddev, no_noise_index=None):
        shape = data.shape
        l = [data]
        for i in range(expand_rate):
            noise = np.random.normal(0, stddev, shape)
            if no_noise_index is not None:
                noise[:, no_noise_index] = 0
            l.append(data+noise)
        return np.concatenate(l, axis=0)

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

        target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
        target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
        target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

        #feature reshape
        #feature reshape: accumulate several(n_step) features into a new feature for the input the lstm
        #target reshape: align the target with the input feature
        ir_train_data = self._feature_reshape(ir_train_raw_data, config.data_step, config.n_step)
        ir_validation_data = self._feature_reshape(ir_validation_raw_data, config.data_step, config.n_step)
        ir_test_data = self._feature_reshape(ir_test_raw_data, config.data_step, config.n_step)

        mete_train_data = self._feature_reshape(mete_train_raw_data, config.data_step, config.n_step)
        mete_validation_data = self._feature_reshape(mete_validation_raw_data, config.data_step, config.n_step)
        mete_test_data = self._feature_reshape(mete_test_raw_data, config.data_step, config.n_step)

        target_train_data = self._target_reshape(target_train_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        target_validation_data = self._target_reshape(target_validation_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)
        target_test_data = self._target_reshape(target_test_raw_data, config.data_step, config.n_step, config.h_ahead, config.n_target)

        #CAUTIOUS: the length of the ir_ttain_data and target_train_data may be differnet
        #the length of mete_test_data may be more short
        #and thus we must use the target data to compute the number

        #assert the the data have been aggregate to the same length
        #there is assertion to this property in this code
        #so you need add some code if the assertion fails
        assert len(ir_train_data) == len(target_train_data)
        assert len(ir_validation_data) == len(target_validation_data)
        assert len(ir_test_data) == len(target_test_data)

        train_index = self._get_valid_index(ir_train_data, mete_train_data, target_train_data)
        validation_index = self._get_valid_index(ir_validation_data, mete_validation_data, target_validation_data)
        test_index = self._get_valid_index(ir_test_data, mete_test_data, target_test_data)

        self.ir_train_data = ir_train_data[train_index]
        self.ir_validation_data = ir_validation_data[validation_index]
        self.ir_test_data = ir_test_data[test_index]

        self.mete_train_data = mete_train_data[train_index]
        self.mete_validation_data = mete_validation_data[validation_index]
        self.mete_test_data = mete_test_data[test_index]

        self.target_train_data = target_train_data[train_index]
        self.target_validation_data = target_validation_data[validation_index]
        self.target_test_data = target_test_data[test_index]

        print len(self.ir_train_data)
        print len(self.target_train_data)

        #add some noise
        #since some features are always a unchanged data which stddve is zero
        #for there field, it do NOT need to add noise to them
        expand_rate = 0
        no_noise_index = (np.std(self.ir_train_data, axis=0) == 0)
        self.ir_train_data = self._add_noise_to_data(self.ir_train_data, expand_rate, 30.0, no_noise_index)

        no_noise_index = (np.std(self.mete_train_data, axis=0) == 0)
        self.mete_train_data = self._add_noise_to_data(self.mete_train_data, expand_rate, 30.0, no_noise_index)

        self.target_train_data = self._add_noise_to_data(self.target_train_data, expand_rate, 0.001)

        #concatenate all valid data
        ir_valid_data = np.concatenate((self.ir_train_data, self.ir_validation_data, self.ir_test_data), axis=0)
        mete_valid_data = np.concatenate((self.mete_train_data, self.mete_validation_data, self.mete_test_data), axis=0)

        #feature scale
        ir_mean = np.mean(ir_valid_data, axis=0)
        ir_std = np.std(ir_valid_data, axis=0)
        self.ir_train_data = (self.ir_train_data - ir_mean) / ir_std
        self.ir_validation_data = (self.ir_validation_data - ir_mean) / ir_std
        self.ir_test_data = (self.ir_test_data - ir_mean) / ir_std

        mete_mean = np.std(mete_valid_data, axis=0)
        mete_std = np.std(mete_valid_data, axis=0)
        mete_std[mete_std==0] = 1.0
        self.mete_train_data = (self.mete_train_data - mete_mean) / mete_std
        self.mete_validation_data = (self.mete_validation_data - mete_mean) / mete_std
        self.mete_test_data = (self.mete_test_data - mete_mean) / mete_std

        #CAUTIOUS: the length of the ir_ttain_data and target_train_data may be differnet
        #the length of mete_test_data may be more short
        #and thus we must use the target data to compute the number
        self.train_num = len(self.target_train_data)
        self.validation_num = len(self.target_validation_data)
        self.test_num = len(self.target_test_data)

        self.batch_size = config.batch_size

        #print the dataset info
        print "Dataset info"
        print "="*80
        print "train number:", self.train_num
        print "validation number:", self.validation_num
        print "test number", self.test_num
        print "batch size:", self.batch_size
        print "use", config.n_step, "hours to predict the next ", config.n_target, " consecutive hours"
        print "\n"

    def next_batch(self):
        """
        @brief return a batch of train and target data
        @return ir_data_batch: [batch_size, n_step, n_input]
        @return mete_data_batch:  [batch_size, n_step, n_input]
        @return target_data_batch: [n_model, batch_size, n_target]
        """
        index = np.random.choice(np.arange(self.train_num), self.batch_size, replace=False)
        ir_batch_data = self.ir_train_data[index]
        mete_batch_data = self.mete_train_data[index]
        target_batch_data = self.target_train_data[index]

        return ir_batch_data, \
                mete_batch_data, \
                target_batch_data

    def get_train_set(self):
        """
        @brief return the total dataset
        """
        return self.ir_train_data, \
                self.mete_train_data, \
                self.target_train_data

    #The returned validataion and test set:
    #ir_data and mete_data: [batch_size, n_step, n_input], batch_size = validation_num/test_num
    #target_data: [batch_size, n_model], each of the target_data contains all model target in a tesor
    def get_validation_set(self):
        """
        @brief return the total validation dataset
        """
        return self.ir_validation_data, \
                self.mete_validation_data, \
                self.target_validation_data

    def get_test_set(self):
        """
        @brief return a test set in the specific test num
        @param test_num The number of test set to return
        """
        return self.ir_test_data, \
                self.mete_test_data, \
                self.target_test_data
