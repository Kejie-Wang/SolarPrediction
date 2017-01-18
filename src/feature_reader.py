import numpy as np

MISSING_VALUE = -99999

class Feature_Reader:
    def _feature_reshape(self, features, n_step, data_step):
        """
        @brief aggregate the multiple features (a successive time features) as a new feature to predict the following time target
        @param features The input features in shape [features_num, feature_dim], each of which is a feature of a specific time
        @param data_step The interval between two new features
        @param n_step The length of the time in the lstm
        @return a new batch shaped features in shape [new_feature_num, n_step, feature_dim]
        """
        shape_features = []
        for ptr in range(n_step, len(features)+1, data_step):
            shape_features.append(features[ptr - n_step:ptr])
        return np.array(shape_features)

    def _read_feature_data(self, data_path, n_step, data_step, shift_day):
        raw_data = np.loadtxt(data_path, delimiter=',', ndmin=2)
        return self._feature_reshape(raw_data, n_step, data_step)[shift_day:]

    def _read_dataset_feature(self, train_path, validation_path, test_path, n_step, data_step, shift_day):
        return self._read_feature_data(train_path, n_step, data_step, shift_day), \
               self._read_feature_data(validation_path, n_step, data_step, shift_day), \
               self._read_feature_data(test_path, n_step, data_step, shift_day)

    def _get_valid_index(self, feature):
        """
        @brief get the valida index of the features since there are some missing value in some feature and target
        @param ir_features, mete_features, sky_cam_features: the three kinds of feautures
        @return Return a indices indicates the valid features (no missing value) index
        """
        num = len(feature)
        missing_index = []
        for i in range(num):
            if (True in np.isnan(feature[i])) or MISSING_VALUE in feature[i]:
                missing_index.append(i)

        return np.setdiff1d(np.arange(num), np.array(missing_index))

    def __init__(self, train_path, validation_path, test_path, n_step, data_step, shift_day):
        #load the data and feature reshape
        #feature reshape: accumulate several(n_step) features into a new feature for the input the lstm
        self.train_data, self.validation_data, self.test_data = self._read_dataset_feature(train_path, validation_path, test_path, n_step, data_step, shift_day)
        #get the valid data index
        self.train_index = self._get_valid_index(self.train_data)
        self.validation_index = self._get_valid_index(self.validation_data)
        self.test_index = self._get_valid_index(self.test_data)

    def get_data(self, train_index, validation_index, test_index):
        self.train_data = self.train_data[train_index]
        self.validation_data = self.validation_data[validation_index]
        self.test_data = self.test_data[test_index]

        #feature scale
        data = np.concatenate((self.train_data, self.validation_data, self.test_data), axis=0)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std==0] = 1.0

        self.mean = mean
        self.std = std

        return self.train_data, self.validation_data, self.test_data

    def get_mean_std(self):
        return self.mean, self.std

    def get_index(self):
        return self.train_index, self.validation_index, self.test_index
