import datetime
import numpy as np

'''
This file used to generate a file contains each hourly data to devote the sky image data
The is three type of data:
existed data: use its time to devote (integer) e.g. 2001010105 devote the image in 2000.01.01 05:00 exists
missing data: the data between 05:00 and 19:00 which is not existed are devoted by MISSING_VALUE
not captured data: since the sky image only captured between 05:00 and 19:00 and the other images are devoted by PAD_VALUE
'''

raw_data_path = '../raw_data/'
exist_image_list_path = '../exist_image_list.csv'

MISSING_VALUE = -99999
PAD_VALUE = -11111

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
            pad_data.append(PAD_VALUE)
        else:
            name = str(it.year).rjust(4, '0') + str(it.month).rjust(2, '0') + str(it.day).rjust(2, '0') + str(hour).rjust(2, '0')
            path = raw_data_path + str(it.year).rjust(4, '0') + '/' + str(it.month).rjust(2, '0') + '/' + str(it.day).rjust(2, '0') + '/' + name + '00.jpg'
            if int(name) in exist_image_list:
                pad_data.append(int(name))
            else:
                pad_data.append(MISSING_VALUE)
    it = it + one_day

np.savetxt('pad_data_path.csv', pad_data, delimiter=',', fmt='%d')
