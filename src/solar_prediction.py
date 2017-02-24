# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import os
import tensorflow as tf
import numpy as np
from config import Model_Config
from reader import Reader
from model import Model
from util import MSE_And_MAE
import time
import sys

model_path = '../saved_model_1/'
output_path = '../output_model_1/'

def fill_feed_dict(x_ir_placeholder, \
                x_mete_placeholder, \
                x_sky_cam_placeholder, \
                hour_index_placeholder, \
                y_placeholder, \
                keep_prob_placeholder, \
                feed_data, keep_prob, modality):
    feed_dict = {}
    index = 0
    if modality[0] == 1:
        feed_dict[x_ir_placeholder] = feed_data[index]
        index += 1
    if modality[1] == 1:
        feed_dict[x_mete_placeholder] = feed_data[index]
        index += 1
    if modality[2] == 1:
        feed_dict[x_sky_cam_placeholder] = feed_data[index]
        index += 1
    feed_dict[hour_index_placeholder] = feed_data[index]
    feed_dict[y_placeholder] = feed_data[-1]
    feed_dict[keep_prob_placeholder] = keep_prob

    return feed_dict

def do_eval(sess,
            evaluation,
            feed_dict):

    return sess.run(evaluation, feed_dict=feed_dict)


def make_folder_path(config, path):
    """
    @brief make the saved model or the result output folder
            in fmt: regressor/modality/n_step+n_shift/
    """
    if os.path.exists(path):
        os.mkdir(path)
    path += str(config.data_step) + '_data_step'
    if not os.path.exists(path):
        os.mkdir(path)
    path += config.regressor + "/"
    if not os.path.exists(path):
        os.mkdir(path)

    hour_size_shift_size = ""
    if config.modality[0] == 1:
        path += "ir"
        hour_size_shift_size += str(config.n_step_1) + "hours" + str(config.n_shift_1) + "shift_"
    if config.modality[1] == 1:
        path += "mete"
        hour_size_shift_size += str(config.n_step_2) + "hours" + str(config.n_shift_2) + "shift_"
    if config.modality[2] == 1:
        path += "skycam"
        hour_size_shift_size += str(config.n_step_3) + "hours" + str(config.n_shift_3) + "shift_"
    path += "/"
    if not os.path.exists(path):
        os.mkdir(path)
    path += hour_size_shift_size + "/"
    if not os.path.exists(path):
        os.mkdir(path)

    path += get_file_name(config) + "/"
    if not os.path.exists(path):
        os.mkdir(path)

    return path

def get_file_name(config):
    """
    @brief get the saved model or result output file name
            in fmt: h_ahead + n_target + hidden_size + quantile_rate(quantile regressor only)
    """
    file_name = ""
    file_name += str(config.h_ahead) + "_ahead" + str(config.n_target) + "_target_"
    if config.modality[0] == 1:
        file_name += str(config.n_first_hidden) + "_"
    if config.modality[1] == 1:
        file_name += str(config.n_second_hidden) + "_"
    if config.modality[2] == 1:
        file_name += str(config.n_third_hidden) + "_"
    if config.regressor == "quantile":
        file_name += str(config.quantile_rate)

    if file_name[-1] == "_":
        file_name = file_name[0:-1]

    return file_name

def main(_):

    #get the config
    config = Model_Config()

    restore = 1
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i].split('=')
        if hasattr(config, arg[0]):
            if arg[0] == "regressor":
                setattr(config, arg[0], arg[1])
            elif arg[0] == "lr" or arg[0]=="quantile_rate":
                setattr(config, arg[0], float(arg[1]))
            else:
                setattr(config, arg[0], int(arg[1]))
        elif arg[0] == "restore":
            restore = int(arg[1])

    n_step_1 = config.n_step_1
    n_step_2 = config.n_step_2
    n_step_3 = config.n_step_3

    n_target = config.n_target

    modality = config.modality

    n_input_ir = config.n_input_ir
    n_input_mete = config.n_input_mete
    width_image = config.width
    height_image = config.height

    epoch_size = config.epoch_size
    print_step = config.print_step

    #define the input and output
    x_ir = tf.placeholder(tf.float32, [None, n_step_1, n_input_ir])
    x_mete = tf.placeholder(tf.float32, [None, n_step_2, n_input_mete])
    x_sky_cam = tf.placeholder(tf.float32, [None, n_step_3, height_image, width_image])
    hour_index = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32, [None, n_target])

    reader = Reader(config)

    model = Model([x_ir, x_mete, x_sky_cam], y_, hour_index, keep_prob, config)

    #new a saver to save the model
    saver = tf.train.Saver()

    saved_model_folder_path = make_folder_path(config, model_path)
    saved_output_folder_path = make_folder_path(config,  output_path)
    file_name = get_file_name(config)

    print '\033[1;31;40m'
    print "save the model in", saved_model_folder_path
    print "save the test result in", saved_output_folder_path
    print "save the file with", file_name
    print '\033[0m'

    validation_min = float('inf')
    best_test_result = None
    lr = config.lr

    tensor_config = tf.ConfigProto()
    tensor_config.gpu_options.allow_growth = True
    with tf.Session(config=tensor_config) as sess:
        # initialize all variables
        tf.global_variables_initializer().run()

        #restore the model if there is a backup of the model
        path = tf.train.latest_checkpoint(saved_model_folder_path)
        if restore == 1 and (not (path is None)):
            save_path = saver.restore(sess, path)
            print '\033[1;34;40m', "restore model", '\033[0m'

        # get the train, validation and test set
        train_set = reader.get_train_set()
        validation_set = reader.get_validation_set()
        test_set = reader.get_test_set()

        # fill the train, validation and test feed dict
        # train_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, hour_index, y_, keep_prob, train_set, 1.0, modality)
        validation_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, hour_index, y_, keep_prob, validation_set, 1.0, modality)
        test_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, hour_index, y_, keep_prob, test_set, 1.0, modality)

        # save the target data, hour_index and missing index
        np.savetxt(saved_output_folder_path + file_name + ".target", test_set[-1], fmt="%.4f", delimiter=',')
        np.savetxt(saved_output_folder_path + file_name + "_hour_index.csv", test_set[-2], fmt="%d", delimiter=',')
        test_missing_index = reader.test_missing_index
        np.savetxt(saved_output_folder_path + file_name + "_missing_index.csv", test_missing_index, fmt="%d", delimiter=',')

        for i in range(epoch_size):
            # do test
            if i%config.test_step == 0:
                #calculate the rmse and mae
                print '\033[1;31;40m'
                if config.regressor == "mse" or config.regressor == "msvr" or config.regressor == "mcc":
                    rmse, mae = do_eval(sess, [model.rmse, model.mae], test_feed)
                    print "Test  RMSE: ", rmse, "Test  MAE: ", mae
                    rmse, mae = do_eval(sess, [model.rmse, model.mae], validation_feed)
                    print "Valid RMSE: ", rmse, "Valid MAE: ", mae

                    mae = 0; num = 0; bs = 2000
                    for ptr in range(bs, len(train_set[-1]), bs):
                        batch = []
                        for j in range(len(train_set)):
                            batch.append(train_set[j][ptr-bs:ptr])
                        train_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, hour_index, y_, keep_prob, batch, 1.0, modality)
                        mae += do_eval(sess, model.mae, train_feed)
                        num += 1
                    print "Train MAE: ", mae / num

                    print "sum of w:", sess.run(model.w_sum)
                    print "bias of regression:", sess.run(model.bias)
                elif config.regressor == "quantile":
                    coverage_rate = do_eval(sess, model.coverage_rate, test_feed)
                    print "Test  covarage rate:", coverage_rate
                    coverage_rate = do_eval(sess, model.coverage_rate, validation_feed)
                    print "Valid covarage rate:", coverage_rate

                    coverage_rate = 0; num = 0; bs = 2000
                    for ptr in range(bs, len(train_set[-1]), bs):
                        batch = []
                        for j in range(len(train_set)):
                            batch.append(train_set[j][ptr-bs:ptr])
                        train_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, hour_index, y_, keep_prob, batch, 1.0, modality)
                        coverage_rate += do_eval(sess, model.coverage_rate, train_feed)
                        num += 1
                    print "Train covarage rate:", coverage_rate / num
                    # coverage_rate = do_eval(sess, model.coverage_rate, train_feed)
                    # print "Train covarage rate:", coverage_rate
                    print "sum of w: ", sess.run(model.w_sum)
                    print "bias of regression: ", sess.run(model.bias)
                print '\033[0m'
                # print do_eval(sess, model.test, test_feed)

            # train
            batch = reader.next_batch()
            feed_dict = fill_feed_dict(x_ir, x_mete, x_sky_cam, hour_index, y_, keep_prob, batch, 0.6, modality)
            sess.run(model.optimize, feed_dict=feed_dict)

            # print step
            if i%config.print_step == 0:
                feed_dict = fill_feed_dict(x_ir, x_mete, x_sky_cam, hour_index, y_, keep_prob, batch, 1.0, modality)
                print '\033[1;32;40m', "Step", i, "train loss:",do_eval(sess, model.loss, feed_dict), '\033[0m'

            if i%10 ==0:
                # validation
                #compare the validation with the last loss
                if config.regressor == "mse" or config.regressor == "msvr" or config.regressor == "mcc":
                    validation_rmse = do_eval(sess, model.rmse, validation_feed)
                    if validation_rmse < validation_min:
                        validation_min = validation_rmse
                        best_test_result = do_eval(sess, model.prediction, test_feed)
                        np.savetxt(saved_output_folder_path + file_name + ".res", best_test_result, fmt="%.4f", delimiter=',')
                        if i>1000:
                            save_path = saver.save(sess, saved_model_folder_path + "model.ckpt")
                            print '\033[1;34;40m', "save the model", '\033[0m'
                elif config.regressor == "quantile":
                    validation_coverage_rate= do_eval(sess, model.coverage_rate, validation_feed)
                    if abs(validation_coverage_rate - config.quantile_rate) < validation_min:
                        validation_min = abs(validation_coverage_rate - config.quantile_rate)
                        best_test_result = do_eval(sess, model.prediction, test_feed)
                        np.savetxt(saved_output_folder_path + file_name + ".res", best_test_result, fmt="%.4f", delimiter=',')

                        if i>500:
                            save_path = saver.save(sess, saved_model_folder_path + "model.ckpt")
                            print '\033[1;34;40m', "save the model", '\033[0m'

if __name__ == "__main__":
    tf.app.run()
