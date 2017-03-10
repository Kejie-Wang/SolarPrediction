import numpy as np

window_size = 1
similar_day_num = 1

irradiance_missing_day_index = [1945]

similar_day_train_data_path = './train/similar_day_train_data_' + str(window_size) + '.csv'
similar_day_validation_data_path = './validation/similar_day_validation_data_' + str(window_size) + '.csv'
similar_day_test_data_path = './test/similar_day_test_data_' + str(window_size) + '.csv'

def similar_day_selection(data, similar_day_num):
    sdata = np.zeros(shape=[data.shape[0], similar_day_num])
    for day in range(len(data)):
        k = 0; j =0
        while j<data.shape[1]:
            if not (data[day][j] in irradiance_missing_day_index):
                sdata[day][k] = data[day][j]
                k += 1
                if k == similar_day_num:
                    break
            j += 1

    return sdata

train_data = np.loadtxt(similar_day_train_data_path, delimiter=',', dtype=np.int)
validation_data = np.loadtxt(similar_day_validation_data_path, delimiter=',', dtype=np.int)
test_data = np.loadtxt(similar_day_test_data_path, delimiter=',', dtype=np.int)

# the first row of train set similar day data is itself and filter it
train_data = train_data[:, 1:]

train_data = similar_day_selection(train_data, similar_day_num)
validation_data = similar_day_selection(validation_data, similar_day_num)
test_data = similar_day_selection(test_data, similar_day_num)

np.savetxt('./train/similar_day_train_data.csv', train_data, delimiter=',', fmt='%d')
np.savetxt('./validation/similar_day_validation_data.csv', validation_data, delimiter=',', fmt='%d')
np.savetxt('./test/similar_day_test_data.csv', test_data, delimiter=',', fmt='%d')
