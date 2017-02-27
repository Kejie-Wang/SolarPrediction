import numpy as np
import cv2

sky_cam_raw_data_path = './raw_data/'
start = 200011
end = 201205

def name2path(root_dir, filename):
    y = filename[:4]
    m = filename[4:6]
    d = filename[6:8]
    path = root_dir + str(y) + '/' + str(m) + '/' + str(d) + '/' + str(filename) + '00.jpg'
    return path

def read_all_image():
    '''
    read all existed images and save them in an file
    '''
    exist_image = np.loadtxt('exist_image_list.csv', dtype='str', delimiter=',')
    data = []
    names = []

    mean = np.load('mean.npy')
    std = np.load('std.npy')
    result = []
    for filename in exist_image:
        # print name
        y = int(filename[:4])
        m = int(filename[4:6])
        d = int(filename[6:8])

        if y*100 + m >= start and y*100 + m <= end:
            print filename
            img = cv2.imread(name2path(sky_cam_raw_data_path, filename), 0)
            img = (img - mean) / std
            data.append(cv2.resize(img, (64,64)))
            names.append(int(filename))

    np.save('./input_data/all_image_gray_64.npy', np.array(data))
    np.savetxt('./input_data/sky_cam_image_name.csv', np.array(names), fmt='%d')

if __name__ == '__main__':
    read_all_image()
