import numpy as np

MISSING_VALUE = -99999

class Target_Reader:
    def _target_reshape(self, targets, max_shift, h_ahead, data_step, n_target):
        """
        @brief aggregate the multiple features as a new target for multiple output and synchronize the target with the features
        @param targets The input targets in shape [target_num, target_dim] ##now target_num=1
        @param n_step  The interval between two new features
        @param h_ahead The interval of the feature and target
        @param n_target The  number of target for a prediction
        @return an aggregated and synchronized targets in shape [new_target_num, n_target, target_dim]
        """
        shape_targets = []
        for ptr in range(max_shift + h_ahead + n_target, len(targets)+1, data_step):
            shape_targets.append(targets[ptr - n_target:ptr])
        return np.array(shape_targets)

    def _read_target_data(self, data_path, max_shift, h_ahead, data_step, n_target):
        raw_data = np.loadtxt(data_path, delimiter=',', dtype=self.dtype)
        return self._target_reshape(raw_data, max_shift, h_ahead, data_step, n_target)

    def _read_dataset_target(self, train_path, validation_path, test_path, max_shift, h_ahead, data_step, n_target):
        return self._read_target_data(train_path, max_shift, h_ahead, data_step, n_target), \
               self._read_target_data(validation_path, max_shift, h_ahead, data_step, n_target), \
               self._read_target_data(test_path, max_shift, h_ahead, data_step, n_target)

    def _get_valid_index(self, target):
        """
        @brief get the valida index of the features since there are some missing value in some feature and target
        @param ir_features, mete_features, sky_cam_features: the three kinds of feautures
        @return Return a indices indicates the valid features (no missing value) index
        """
        num = len(target)
        missing_index = []
        for i in range(num):
            if (True in np.isnan(target[i])) or MISSING_VALUE in target[i]:
                missing_index.append(i)

        return np.setdiff1d(np.arange(num), np.array(missing_index))

    def __init__(self, train_path, validation_path, test_path, max_shift, h_ahead, data_step, n_target, dtype=np.float32):
        self.dtype = dtype
        self.train_data, self.validation_data, self.test_data = self._read_dataset_target(train_path, validation_path, test_path, max_shift, h_ahead, data_step, n_target)

        #get the valid data index
        self.train_index = self._get_valid_index(self.train_data)
        self.validation_index = self._get_valid_index(self.validation_data)
        self.test_index = self._get_valid_index(self.test_data)

    def get_data(self, train_index, validation_index, test_index):
        self.train_data = self.train_data[train_index]
        self.validation_data = self.validation_data[validation_index]
        self.test_data = self.test_data[test_index]

        return self.train_data, self.validation_data, self.test_data

    def get_index(self):
        return self.train_index, self.validation_index, self.test_index
