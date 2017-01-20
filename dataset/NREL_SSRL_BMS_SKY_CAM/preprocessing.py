#-*- coding: utf-8 -*-
'''
According to the paper, the authors extracted upto 80 frames from each video,
they did not mention if they grabbed first 80 frames, or sampled 80 frames with same intervals,
but anyway I did the latter.
'''
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import os
import numpy as np
import pandas as pd
from skimage import feature
import scipy
import math
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import mean_squared_error
import multiprocessing
#from cnn_util import *

# raw_image_path = '/home/lcc/code/data/SSRL_SKY_CAM_IMAGE/'
# input_data_path = '/home/lcc/code/python/SolarPrediction/dataset/NREL_SSRL_BMS_SKY_CAM/input_data/'
raw_image_path = '/media/lcc/Windows/Downloads/SSRL_SKY/'
input_data_path = '/home/lcc/code/python/SolarPrediction/dataset/NREL_SSRL_BMS_SKY_CAM/input_data/'
year = range(1999, 2017)
def preprocess_frame(image, target_height=224, target_width=224):

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def pad_data(method, feat_size):
    new_data = []
    start_day = '20080101'
    end_day = '20160731'
    day_list = np.loadtxt('day_list.csv', dtype='str')
    date = np.loadtxt('exist_image_list.csv', dtype='str')
    feat = np.loadtxt(input_data_path + 'raw_' + method + '_' + str(feat_size) + '.csv', dtype='float')
    #data = np.loadtxt(path + method + '.csv', delimiter=',',dtype='str')
    #date = [str(int(float(i))) for i in data[:,0]]
    #feat = np.array(data[:,1:], dtype = 'float')
    #feat = data
    #feat_size = feat.shape[1]
    print feat
    print date
    print day_list
    idx = 0
    for day in day_list:
        hours = range(0,24)
        for hour in hours[:5]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([0] * feat_size)
        for hour in hours[5:-4]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day +  str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            print 'Day_Hour:', day_hour
            print 'Idx:', idx
            print date[idx]
            while date[idx] < day_hour:
                idx += 1
            if date[idx] == day_hour:
                new_data.append(feat[idx])
            else:
                f = np.array([-99999]*feat_size)
                new_data.append(f)
        for hour in hours[-4:]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([0] * feat_size)

    new_data = np.array(new_data)
    print new_data.shape
    np.savetxt(input_data_path + method + '_' + str(feat_size) + '.csv', new_data, fmt='%.4f',delimiter=',')

def pad_data_image_path():
    new_data = []
    start_day = '20080101'
    end_day = '20160731'
    day_list = np.loadtxt('day_list.csv', dtype='str')
    date = np.loadtxt('exist_image_list.csv', dtype='str')
    print date
    print day_list
    idx = 0
    for day in day_list:
        hours = range(0,24)
        for hour in hours[:5]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([-11111])
        for hour in hours[5:-4]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day +  str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            print 'Day_Hour:', day_hour
            print 'Idx:', idx
            print date[idx]
            while date[idx] < day_hour:
                idx += 1
            if date[idx] == day_hour:
                new_data.append([int(date[idx])])
            else:
                new_data.append([-99999])
        for hour in hours[-4:]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([-11111])

    new_data = np.array(new_data)
    print new_data.shape
    np.savetxt('pad_data_path.csv', new_data, fmt='%12.0f', delimiter=',')
# def prerain_CNN():
#     vgg_model = '/home/lcc/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'
#     vgg_deploy = '/home/lcc/caffe/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'
#     year = range(1999,2015)
#
#     #videos = filter(lambda x: x.endswith('avi'), videos)
#     width = 227
#     height = 227
#     print 'Before CNN'
#     cnn = CNN(model=vgg_model, deploy=vgg_deploy, width=width, height=height)
#     print 'After CNN'
#     for y in year[::-1]:
#         print raw_image_path + str(y) + '/'
#         feat_list = []
#         for parent, dirnames, filenames in os.walk(raw_image_path+str(y)+'/'):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
#             for filename in filenames:  # 输出文件信息
#                 if not filename.endswith('.jpg'):
#                     continue
#                 # print "parent is:" + parent
#                 # print "filename is:" + filename
#                 # print "the full name of the file is:" + os.path.join(parent, filename)  # 输出文件路径信息
#                 file = os.path.join(parent, filename)
#                 img_list = []
#                 img = cv2.imread(file)
#                 if img is None:
#                     continue
#                 img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
#                 img_list.append(img)
#                 img_list = np.array(img_list)
#                 feat = cnn.get_features(img_list,layers='fc7',layer_sizes=[4096])[0].tolist()
#                 print filename[:-4]
#                 feat.insert(0, int(filename[:-4]))
#                 feat_list.append(feat)
#         feat_list = np.array(feat_list)
#         print feat_list.shape
#         np.savetxt(input_data_path+str(y)+"_rcnnet.csv", feat_list, delimiter=",")

def Dense_sift(img):
    img = cv2.resize(img,(25,25))
    #dense = cv2.FeatureDetector_create('Dense')
    dense = cv2.FeatureDetector_create('Dense')
    f = '{} ({}): {}'
    for param in dense.getParams():
        type_ = dense.paramType(param)
        if type_ == cv2.PARAM_BOOLEAN:
            print f.format(param, 'boolean', dense.getBool(param))
        elif type_ == cv2.PARAM_INT:
            print f.format(param, 'int', dense.getInt(param))
        elif type_ == cv2.PARAM_REAL:
            print f.format(param, 'real', dense.getDouble(param))
        else:
            print param
    #dense = cv2.setDouble('')
    dense.setDouble('initFeatureScale', 10)
    dense.setDouble('featureScaleMul', 10)
    dense.setInt('initXyStep', 10)
    print type(dense)
    kp = dense.detect(img)
    print len(kp)
    sift = cv2.SIFT()
    kp,des = sift.compute(img,kp)
    print des.ravel().shape
    return des
    #print type(kp[0]), type(des[0])
    #print len(kp), len(des)

def HOG_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray,(64,128))
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    h = h.ravel()
    return h.tolist()


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist
def LBP_feature(img, numPoints = 24, radius = 8):
    desc = LocalBinaryPatterns(numPoints, radius)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    #print hist.shape
    return hist.tolist()

def get_feature(method):
    feat_list = []
    for y in year:
        print raw_image_path + str(y) + '/'
    #    feat_list = []
        for parent, dirnames, filenames in os.walk(raw_image_path + str(y) + '/'):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
            for filename in filenames:  # 输出文件信息
                if not filename.endswith('.jpg'):
                    continue
                file = os.path.join(parent, filename)
                img = cv2.imread(file)
                if img is None:
                    continue
                print filename[:-4]
                if method == 'HOG':
                    feat = HOG_feature(img)
                elif method == 'LBP':
                    feat = LBP_feature(img)
                elif method == 'DSIFT':
                    feat = Dense_sift(img)
                elif method == 'exist_image':
                    feat = int(filename[:-4])
                feat_list.append(feat)
    feat_list = np.array(feat_list)
    print feat_list.shape

    #np.savetxt(input_data_path + 'raw_' + method + '_' + str(feat_list.shape[1]) + '.csv', feat_list, delimiter=",", fmt = '%.4f')
    #np.savetxt('input_data_path.csv', feat_list, delimiter=',',fmt = '%12.0f')

def num_cloud_pixel(img):
    Thr_NC = 0.8
    b, g, r = cv2.split(img)
    RBR = np.true_divide(r, b + 0.001)
    return np.sum([1 if i > Thr_NC else 0 for i in RBR.ravel()])

def gradient_magnitude(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Sobel(img, -1, 1, 1)
    return np.sum(dst.ravel())

def intensity_level(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(img.ravel())

def num_corner(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(img, 7, 7, 0.04)
    return np.sum([1 if i > (0.1 * dst.max()) else 0 for i in dst.ravel()])

def all_sky_image_features(img_lst):
    res = []
    img_lst_gray = []
    idx = 0
    for img in img_lst:
        img_lst_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    '''
     Numbers of cloud pixels
    '''
    NC = []; Thr_NC = 0.8
    for img in img_lst:
        b, g, r = cv2.split(img)
        RBR = np.true_divide(r, b + 0.001)
        NC.append(np.sum([1 if i > Thr_NC else 0 for i in RBR.ravel()]))
    res.append(np.mean(NC))
    res.append(np.var(NC))
    #print 'mean(NC) var(NC)', np.mean(NC), np.var(NC)

    '''
    Frame difference
    '''
    DA = []; DC = []; Thr_DF = 0.1
    for i in range(len(img_lst)):
        if i == 0: continue
        dif = (img_lst[i].ravel() - img_lst[i - 1].ravel())
        tmp = [abs(i) if abs(i) > Thr_DF else 0 for i in dif]
        DA.append(np.sum(tmp))
        tmp = [1 if abs(i) > Thr_DF else 0 for i in dif]
        DC.append(np.sum(tmp))
    res.append(np.mean(DA))
    res.append(np.mean(DC))
    res.append(np.var(DA))
    res.append(np.var(DC))
    #print 'mean(DA) var(DA)', np.mean(DA), np.var(DA)
    #print 'mean(DC) var(DC)', np.mean(DC), np.var(DC)

    '''
    Gradient magnitude
    '''
    GM = []
    for img in img_lst_gray:
        dst = cv2.Sobel(img, -1, 1, 1)
        GM.append(np.sum(dst.ravel()))
    res.append(np.mean(GM))
    res.append(np.var(GM))
    #print 'mean(GM) var(GM)', np.mean(GM), np.var(GM)

    '''
    Intensity Level
    '''
    IL = []
    for img in img_lst_gray:
        IL.append(np.mean(img.ravel()))
    res.append(np.mean(IL))
    res.append(np.var(IL))
    #print 'mean(IL) var(IL)', np.mean(IL), np.var(IL)

    '''
    Accumulated intensity along the vertical line of sum
    '''
    # for img in img_lst_gray:
    #     dst = (cv2.HoughLines(img, 1, math.pi / 180, 125))[0]
    #     ang = [i if i >= 0 and i <= math.pi / 2 else abs(math.pi - i) for i in dst[:,1]]
    #     ndst = np.array([i if i[1] < 1 else None for i in dst])
    #     idx = np.argsort(ang)
    #     print ndst
    #     #print dst[idx]

    '''
    Number of Corners
    '''
    COR = []
    for img in img_lst_gray:
        dst = cv2.cornerHarris(img, 7, 7, 0.04)
        COR.append(np.sum([1 if i > (0.1 * dst.max()) else 0 for i in dst.ravel()]))
    res.append(np.mean(COR))
    res.append(np.var(COR))
    #print 'mean(COR) var(COR)', np.mean(COR), np.var(COR)

    return res

def automatic_cloud(img):
    b, g, r = cv2.split(img)
    '''
    Spectral feature
    '''
    #Mean(R and B)
    ME_r = np.mean(r.ravel())
    ME_b = np.mean(b.ravel())
    ME_g = np.mean(g.ravel())


    #Standard deviation(B)
    SD_b = np.std(b.ravel())

    #Skewness(B)
    SK_b = scipy.stats.skew(b.ravel())

    #Difference(R-G, R-B and G-B)
    D_rb = ME_r - ME_b
    D_rg = ME_r - ME_g
    D_gb = ME_g - ME_b

    '''
    Textural features(Based on GCLM)
    '''
    gclm = gclm_matrix(img)
    #Engergy(B)
    EN_b = np.sum(gclm.ravel())

    #Entropy(B)
    ENT_b = 0
    for i in gclm.ravel():
        if i <= 0.0: continue
        ENT_b += i * math.log(i,2)

    #Contrast(B)
    CON_b = 0
    for i in range(256):
        for j in range(256):
            CON_b += (i-j)**2 * gclm[i,j]

    #Homogenity(B)
    HOM_b = 0
    for i in range(256):
        for j in range(256):
            HOM_b += gclm[i,j] / (1 + abs(i-j))

    res = [ME_r,ME_g, ME_b, SD_b, SK_b, D_gb, D_rb, D_rg, EN_b, ENT_b, CON_b, HOM_b, num_cloud_pixel(img), num_corner(img), intensity_level(img), gradient_magnitude(img)]
    #print res
    return res

def gclm_matrix(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gclm = feature.greycomatrix(img,[2],[np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
    m = None
    for i in range(4):
        if m is None:
            m = gclm[:,:,0,i]
        m = gclm[:,:,0,i] + m
    m = np.true_divide(m, 4)
    return m


def name2path(root_dir, filename):
    y = filename[:4]
    m = filename[4:6]
    d = filename[6:8]
    path = root_dir + str(y) + '/' + str(m) + '/' + str(d) + '/' + str(filename) + '.jpg'
    return path

def path2image(self, data, index, add_noise = False, image_scale = False):
    mean = cv2.resize(np.load('mean.npy'), (self.heigth, self.width))
    std = cv2.resize(np.load('std.npy'), (self.heigth, self.width))
    img_list = []
    for idx in index:
        imgs = []
        for i in range(self.n_step):
            if data[idx, i] == -11111:
                imgs.append(np.zeros((self.heigth,self.width, 3), dtype='uint8'))
            else:
                filename = str(int(data[idx, i]))
                y = filename[:4]
                m = filename[4:6]
                d = filename[6:8]
                path = sky_cam_raw_data_path + str(y) + '/' + str(m) + '/' + str(d) + '/' + str(filename) + '.jpg'
                #tt = cv2.imread(path)
                #print i, filename, tt.dtype
                #cv2.imshow(filename, tt)
                #cv2.waitKey(0)
                tmp = cv2.resize(cv2.imread(path), (self.heigth, self.width))
                # if image_scale is True:
                #     tmp -= mean
                #     tmp /= std
                # mean_noise = self.noise_mean
                # sigma_noise = self.noise_vari
                # noise = np.random.normal(mean_noise, sigma_noise, size=(self.heigth, self.width))
                # if add_noise is True and random.random() < self.noise_ratio:
                #     tmp += noise
                imgs.append(tmp)
        img_list.append(imgs)
    #print img_list
    return np.array(img_list)

def feature_extraction(X_path):
    sky_cam_raw_data_path = '../dataset/NREL_SSRL_BMS_SKY_CAM/SSRL_SKY/'
    pool = multiprocessing.Pool(processes=8)
    result = []
    X = []
    for i in range(len(X_path)) :
        filename = str(int(X_path[i]))
        if filename == '-11111':
            result.append([0]*16)
        elif filename == '-99999':
            result.append([-99999]*16)
        else:
            path = name2path(sky_cam_raw_data_path, filename)
            print path
            img = cv2.imread(path)
            result.append(pool.apply_async(automatic_cloud,(img,)))

    pool.close()
    pool.join()
    print result
    for res in result:
        if isinstance(res,(list)) is True:
            X.append(res)
        else:
            X.append(res.get())
    X = np.array(X)
    return X

def detect_clouds(image):
    b, g, r = cv2.split(image)
    RBR = np.true_divide(r, b + 0.001)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if RBR[i,j] > 0.8 and RBR[i,j] < 1:
                image[i,j,0] = 0;
                image[i,j,1] = 255;
                image[i,j,2] = 0;

    return image

def frame_diff(img_1,img_2):
    img_1_gray = (cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)).astype('int16')
    img_2_gray = (cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)).astype('int16')
    img_dif = np.abs(img_1_gray - img_2_gray)
    img_DA = np.copy(img_dif)
    img_DC = np.copy(img_dif)
    for i in range(img_dif.shape[0]):
        for j in range(img_dif.shape[1]):
            if img_dif[i,j] < 20:
                img_DA[i,j] = 0
                img_DC[i,j] = 0
            else:
                img_DC[i,j] = 255
    img_dif = img_dif.astype('uint8')
    img_DA = img_DA.astype('uint8')
    img_DC = img_DC.astype('uint8')
    return img_DA, img_DC

def read_all_image():
    sky_cam_raw_data_path = 'SSRL_SKY/'
    exist_image = np.loadtxt('exist_image_list.csv', dtype='str', delimiter=',')
    res = []
    mean = np.load('mean.npy')
    std = np.load('std.npy')
    result = []
    for name in exist_image:
        print name
        img = cv2.imread(name2path(sky_cam_raw_data_path, name),0)
        img = (img - mean) / std
        res.append(cv2.resize(img, (64,64)))
    np.save('all_image_gray_64.npy', np.array(res))

if __name__ == '__main__':
    read_all_image()
    # X_validation_path = np.loadtxt('../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/validation/sky_cam_validation_data.csv', dtype='float', delimiter=',')
    # X_validation = feature_extraction(X_validation_path)
    # np.save('automatic_cloud_X_validation', X_validation)
    #
    # X_train_path = np.loadtxt('../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/train/sky_cam_train_data.csv', dtype='float', delimiter=',')
    # X_train = feature_extraction(X_train_path)
    # np.save('automatic_cloud_X_train', X_train)
    #
    # X_test_path = np.loadtxt('../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/test/sky_cam_test_data.csv', dtype='float', delimiter=',')
    # X_test = feature_extraction(X_test_path)
    # np.savetxt('automatic_cloud_X_testcsv', X_test)

    # X_path = np.loadtxt('../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/pad_data_path.csv', dtype='float', delimiter=',')
    # X = feature_extraction(X_path)
    # np.savetxt('automatic_cloud_X.csv', X, delimiter=',')

    #sky_cam_raw_data_path = '/home/lcc/code/data/SSRL_SKY_CAM_IMAGE/'
    # X_train_path = np.loadtxt('../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/train/sky_cam_train_data.csv', dtype='float', delimiter=',')
    #
    # y_train_path = np.loadtxt('../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv', dtype='float', delimiter=',')
    #
    # X_test_path = np.loadtxt('../dataset/NREL_SSRL_BMS_SKY_CAM/input_data/test/sky_cam_test_data.csv', dtype='float', delimiter=',')
    # y_test_path = np.loadtxt('../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv', dtype='float', delimiter=',')


    # X_test = []
    # y_test = []
    # print 'Test'
    # for i in range(len(X_test_path)):
    #     filename = str(int(X_test_path[i]))
    #     if filename == '-11111' or filename == '-99999':
    #         continue
    #     path = name2path(sky_cam_raw_data_path, filename)
    #     img = cv2.imread(path)
    #     print 'Test' + path
    #     result.append(pool.apply_async(automatic_cloud, (img,)))
    #     y_test.append(y_test_path[i])


    # for res in result:
    #     print 'Idx:', idx
    #     if idx < len(y_train):
    #         print res.get()
    #         X_train.append(res.get())
    #     else:
    #         X_test.append(res.get())
    #     idx += 1
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)
    # print X_train.shape, y_train.shape
    # print X_test.shape, y_test.shape

    # np.save('automatic_cloud_X_train', X_train)
    # np.save('automatic_cloud_X_test', X_test)
    # np.save('automatic_cloud_y_train', y_train)
    # np.save('automatic_cloud_y_test', y_test)
    # est = rf(n_estimators = 1000, n_jobs = 8).fit(X_train, y_train)
    # print mean_squared_error(y_test, est.predict(X_test))
    # print est.feature_importances_
