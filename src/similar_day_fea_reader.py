import numpy as np

HOUR_IN_A_DAY = 24

class Similar_Day_Reader:

    def _get_day_tuple(self, time):
        print time.shape
        time = time[:, 0:3]
        return [tuple(x) for x in time]

    def __init__(self, similar_day_train_data_path, similar_day_validation_data_path, similar_day_test_data_path, time_train_data_path, time_validation_data_path, time_test_data_path):
        train_data = np.loadtxt(similar_day_train_data_path, delimiter=',')
        validation_data = np.loadtxt(similar_day_validation_data_path, delimiter=',')
        test_data = np.loadtxt(similar_day_test_data_path, delimiter=',')

        time_train_data = np.loadtxt(time_train_data_path, delimiter=',', dtype=np.int)
        time_validation_data = np.loadtxt(time_validation_data_path, delimiter=',', dtype=np.int)
        time_test_data = np.loadtxt(time_test_data_path, delimiter=',', dtype=np.int)

        # data: [day_num, n_step] ====> [day_num, n_step, 1]
        data = np.concatenate((train_data, validation_data, test_data), axis=0)
        data = np.reshape(data, [data.shape[0], data.shape[1], 1])

        time = np.concatenate((time_train_data, time_validation_data, time_test_data), axis=0)
        time = np.reshape(time, [-1, HOUR_IN_A_DAY, time.shape[-1]])
        time = time[:,0, :]

        time = self._get_day_tuple(time)

        self.dic = dict(zip(time, data))

    def get_day_fea_map(self):
        return self.dic

    def get_data(self, train_time, validation_time, test_time):
        train_time = self._get_day_tuple(train_time)
        validation_time = self._get_day_tuple(validation_time)
        test_time = self._get_day_tuple(test_time)

        return np.array([self.dic[x] for x in train_time]), np.array([self.dic[x] for x in validation_time]), np.array([self.dic[x] for x in test_time])
