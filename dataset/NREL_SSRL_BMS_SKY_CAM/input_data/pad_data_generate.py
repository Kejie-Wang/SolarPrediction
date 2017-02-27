import datetime
import numpy as np

raw_data_path = '../raw_data/'
exist_image_list_path = '../exist_image_list.csv'

MISSING_VALUE = -99999


year_start = 2000
month_start = 11
year_end = 2012
month_end = 6

exist_image_list = np.loadtxt(exist_image_list_path, delimiter=',', dtype=np.int)
exist_image_list = set(exist_image_list)

end_day = datetime.datetime(year_end, month_end, 1)
it = datetime.datetime(year_start, month_start, 1)

pad_data = []
one_day = datetime.timedelta(days=1)

while it < end_day:
    print it
    for hour in range(0, 24):
        if hour < 5 or hour > 19:
            pad_data.append(-11111)
        else:
            name = str(it.year).rjust(4, '0') + str(it.month).rjust(2, '0') + str(it.day).rjust(2, '0') + str(hour).rjust(2, '0')
            path = raw_data_path + str(it.year).rjust(4, '0') + '/' + str(it.month).rjust(2, '0') + '/' + str(it.day).rjust(2, '0') + '/' + name + '00.jpg'
            if int(name) in exist_image_list:
                pad_data.append(int(name))
            else:
                pad_data.append(MISSING_VALUE)
    it = it + one_day

np.savetxt('pad_data_path.csv', pad_data, delimiter=',', fmt='%d')
