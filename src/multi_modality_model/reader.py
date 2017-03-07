# -*- coding: utf-8 -*-
from __future__ import division
import sys
# append the upper dir into the system path
sys.path.append('../')
import numpy as np
from feature_reader import Feature_Reader
from target_reader import Target_Reader
import cv2

HOUR_IN_A_DAY = 24

ir_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/ir_train_data.csv"
mete_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv"
sky_cam_train_data_path = "../../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/train/sky_cam_train_data.csv"
target_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"
time_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/time_train_data.csv"

ir_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/ir_validation_data.csv"
mete_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/mete_validation_data.csv"
sky_cam_validation_data_path = "../../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/validation/sky_cam_validation_data.csv"
target_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"
time_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/time_validation_data.csv"

ir_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/ir_test_data.csv"
mete_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv"
sky_cam_test_data_path = "../../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/test/sky_cam_test_data.csv"
target_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"
time_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/time_test_data.csv"

sky_cam_image_data_path = '../../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/all_image_gray_64.npy'
sky_cam_exist_image_list_path = '../../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/sky_cam_image_name.csv'

class Reader:

    def _get_hour_index_and_filter_data(self, max_shift, h_ahead, data_step, data_index):
        # get hour index
        start_hour_index = (max_shift + h_ahead) % HOUR_IN_A_DAY
        hour_index = np.arange(start_hour_index, max(data_index)*data_step+start_hour_index+1, data_step)
        hour_index = hour_index[data_index] % HOUR_IN_A_DAY
        # filter the data
        index = np.logical_and(hour_index>=5, hour_index<=18)

        return data_index[index], np.reshape(hour_index[index], [-1, 1])

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
        n_step_3 = config.n_step_3

        shift = []
        if self.modality[0] == 1:
            shift.append(config.n_shift_1)
        if self.modality[1] == 1:
            shift.append(config.n_shift_2)
        if self.modality[2] == 1:
            shift.append(config.n_shift_3)
        max_shift = max(shift)

        train_index = []; validation_index = []; test_index = []
        #read first modality data
        if self.modality[0] == 1:
            shift_day_1 = (max_shift - config.n_shift_1) // data_step; assert (max_shift - config.n_shift_1)%data_step == 0
            ir_feature_reader = Feature_Reader(ir_train_data_path, ir_validation_data_path, ir_test_data_path, n_step_1, data_step, shift_day_1)
            ir_train_index, ir_validation_index, ir_test_index = ir_feature_reader.get_index()
            train_index.append(ir_train_index); validation_index.append(ir_validation_index); test_index.append(ir_test_index)

        #read second modality data
        if self.modality[1] == 1:
            shift_day_2 = (max_shift - config.n_shift_2) // data_step; assert (max_shift - config.n_shift_2)%data_step == 0
            mete_feature_reader = Feature_Reader(mete_train_data_path, mete_validation_data_path, mete_test_data_path, n_step_2, data_step, shift_day_2)
            mete_train_index, mete_validation_index, mete_test_index = mete_feature_reader.get_index()
            train_index.append(mete_train_index); validation_index.append(mete_validation_index); test_index.append(mete_test_index)

        #read third modality data
        if self.modality[2] == 1:
            shift_day_3 = (max_shift - config.n_shift_3) // data_step; assert (max_shift - config.n_shift_3)%data_step == 0
            sky_cam_feature_reader = Feature_Reader(sky_cam_train_data_path, sky_cam_validation_data_path, sky_cam_test_data_path, n_step_3, data_step, shift_day_3)
            sky_cam_train_index, sky_cam_validation_index, sky_cam_test_index = sky_cam_feature_reader.get_index()
            train_index.append(sky_cam_train_index); validation_index.append(sky_cam_validation_index); test_index.append(sky_cam_test_index)

            #Read all images into memory
            self.images = np.load(sky_cam_image_data_path)
            #Define a dictionary to store the indexes of images in self.images
            self.file2idx = dict()
            exist_image_list = np.loadtxt(sky_cam_exist_image_list_path, dtype=np.int)
            idx = 0
            for f in exist_image_list:
                self.file2idx[f] = idx
                idx += 1

        # read target data
        target_reader = Target_Reader(target_train_data_path, target_validation_data_path, target_test_data_path, max_shift, h_ahead, data_step, n_target)
        target_train_index, target_validation_index, target_test_index = target_reader.get_index()
        train_index.append(target_train_index); validation_index.append(target_validation_index); test_index.append(target_test_index)

        # read the target time index
        time_reader = Target_Reader(time_train_data_path, time_validation_data_path, time_test_data_path, max_shift, h_ahead, data_step, n_target)

        # get the valid index
        train_index = reduce(np.intersect1d, train_index)
        validation_index = reduce(np.intersect1d, validation_index)
        test_index= reduce(np.intersect1d, test_index)

        # get the hour index
        # filter the hour index less than 5 or bigger than 18
        # the irradiance in this range is nearly zero and it need not forecast
        train_index, self.train_hour_index = self._get_hour_index_and_filter_data(max_shift, h_ahead, data_step, train_index)
        validation_index, self.validation_hour_index = self._get_hour_index_and_filter_data(max_shift, h_ahead, data_step, validation_index)
        test_index, self.test_hour_index = self._get_hour_index_and_filter_data(max_shift, h_ahead, data_step, test_index)

        if self.modality[0] == 1:
            self.ir_train_data, self.ir_validation_data, self.ir_test_data = ir_feature_reader.get_data(train_index, validation_index, test_index)
            self.ir_mean, self.ir_std = ir_feature_reader.get_mean_std()
        if self.modality[1] == 1:
            self.mete_train_data, self.mete_validation_data, self.mete_test_data = mete_feature_reader.get_data(train_index, validation_index, test_index)
            self.mete_mean, self.mete_std = mete_feature_reader.get_mean_std()
        if self.modality[2] == 1:
            self.sky_cam_train_data, self.sky_cam_validation_data, self.sky_cam_test_data = sky_cam_feature_reader.get_data(train_index, validation_index, test_index)

        self.target_train_data, self.target_validation_data, self.target_test_data = target_reader.get_data(train_index, validation_index, test_index)
        self.time_train_data, self.time_validation_data, self.time_test_data = time_reader.get_data(train_index, validation_index, test_index)

        self.test_missing_index = target_reader.get_test_missing_index(test_index)

        #CAUTIOUS: the length of the ir_train_data and target_train_data may be differnet
        #the length of mete_test_data may be more short
        #and thus we must use the target data to compute the number
        self.train_num = len(self.target_train_data)
        self.validation_num = len(self.target_validation_data)
        self.test_num = len(self.target_test_data)

        self.batch_size = config.batch_size

        self.n_step_3 = config.n_step_3
        self.width = config.width
        self.height = config.height

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

    def path2image(self, data):
        """
        data : [batch_size, n_step_3]
        """
        img_list = []
        for idx in range(len(data)):
            img = []
            for i in range(self.n_step_3):
                if int(data[idx, i]) == -11111:
                    img.append(np.zeros((self.height,self.width)))
                else:
                    filename = int(data[idx, i])
                    tmp = self.images[self.file2idx[filename]]
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
        index = np.random.choice(np.arange(self.train_num), self.batch_size, replace=False)
        batch = []
        if self.modality[0] == 1:
            ir_batch_data = (self.ir_train_data[index] - self.ir_mean) / self.ir_std
            batch.append(ir_batch_data)
        if self.modality[1] == 1:
            mete_batch_data = (self.mete_train_data[index] - self.mete_mean) / self.mete_std
            batch.append(mete_batch_data)
        if self.modality[2] == 1:
            sky_cam_batch_data = self.path2image(self.sky_cam_train_data[index])
            batch.append(sky_cam_batch_data)
        hour_index_batch_data = self.train_hour_index[index]
        batch.append(hour_index_batch_data)
        target_batch_data = self.target_train_data[index]
        batch.append(target_batch_data)

        return batch

    def get_train_set(self, start=None, length=None):
        """
        @brief return the total dataset
        """
        if start is None:
            start = 0
        if length is None or start + length > self.train_num:
            length = self.train_num - start

        train_set = []
        if self.modality[0] == 1:
            train_set.append((self.ir_train_data[start:start+length] - self.ir_mean) / self.ir_std)
        if self.modality[1] == 1:
            train_set.append((self.mete_train_data[start:start+length] - self.mete_mean) / self.mete_std)
        if self.modality[2] == 1:
            train_set.append(self.path2image(self.sky_cam_train_data[start:start+length]))
        train_set.append(self.train_hour_index[start:start+length])
        train_set.append(self.target_train_data[start:start+length])

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

        validation_set = []
        if self.modality[0] == 1:
            validation_set.append((self.ir_validation_data[start:start+length] - self.ir_mean) / self.ir_std)
        if self.modality[1] == 1:
            validation_set.append((self.mete_validation_data[start:start+length] - self.mete_mean) / self.mete_std)
        if self.modality[2] == 1:
            validation_set.append(self.path2image(self.sky_cam_validation_data[start:start+length]))
        validation_set.append(self.validation_hour_index[start:start+length])
        validation_set.append(self.target_validation_data[start:start+length])

        return validation_set

    def get_test_set(self, start=None, length=None):
        """
        @brief return a test set in the specific test num
        """
        if start is None:
            start = 0
        if length is None or start + length > self.test_num:
            length = self.test_num - start

        test_set = []
        if self.modality[0] == 1:
            test_set.append((self.ir_test_data[start:start+length] - self.ir_mean) / self.ir_std)
        if self.modality[1] == 1:
            test_set.append((self.mete_test_data[start:start+length] - self.mete_mean) / self.mete_std)
        if self.modality[2] == 1:
            test_set.append(self.path2image(self.sky_cam_test_data[start:start+length]))
        test_set.append(self.test_hour_index[start:start+length])
        test_set.append(self.target_test_data[start:start+length])

        return test_set
