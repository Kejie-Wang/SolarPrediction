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

def fill_feed_dict(x_ir_placeholder, x_mete_placeholder, x_sky_cam_placeholder, y_, keep_prob_placeholder, feed_data, keep_prob, modality):
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
    feed_dict[y_] = feed_data[index]
    feed_dict[keep_prob_placeholder] = keep_prob

    return feed_dict

def do_eval(sess,
            evaluation,
            feed_dict):

    return sess.run(evaluation, feed_dict=feed_dict)

def main(_):
    #get the config
    config = Model_Config()

    n_step = config.n_step
    n_target = config.n_target

    modality = config.modality

    n_input_ir = config.n_input_ir
    n_input_mete = config.n_input_mete
    width_image = config.width
    height_image = config.height

    epoch_size = config.epoch_size
    print_step = config.print_step

    #define the input and output
    x_ir = tf.placeholder(tf.float32, [None, n_step, n_input_ir])
    x_mete = tf.placeholder(tf.float32, [None, n_step, n_input_mete])
    x_sky_cam = tf.placeholder(tf.float32, [None, n_step, height_image, width_image])

    keep_prob = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32, [None, n_target])

    reader = Reader(config)

    model = Model([x_ir, x_mete, x_sky_cam], y_, keep_prob, config)

    #new a saver to save the model
    saver = tf.train.Saver()

    validation_last_loss = float('inf')
    best_test_result = None

    tensor_config = tf.ConfigProto()
    tensor_config.gpu_options.allow_growth = True
    with tf.Session(config=tensor_config) as sess:
        # initialize all variables
        tf.global_variables_initializer().run()

        train_set = reader.get_train_set()
        validation_set = reader.get_validation_set()
        test_set = reader.get_test_set()

        train_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, keep_prob, train_set, 1.0, modality)
        validation_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, keep_prob, validation_set, 1.0, modality)
        test_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, keep_prob, test_set, 1.0, modality)

        np.set_printoptions(precision=4)
        test_target = test_set[-1]
        for i in sorted(test_target):
            print test_target[i]

        for i in range(epoch_size):
            # test
            if i%config.test_step == 0:
                #calculate the mse and mae
                rmse, mae = do_eval(sess, [model.rmse, model.mae], test_feed)
                print "Test  RMSE: ", rmse, "Test  MAE: ", mae
                rmse, mae = do_eval(sess, [model.rmse, model.mae], validation_feed)
                print "Valid RMSE: ", rmse, "Valid MAE: ", mae
                rmse, mae = do_eval(sess, [model.rmse, model.mae], train_feed)
                print "Train RMSE: ", rmse, "Train MAE: ", mae
                print "sum of w: ", sess.run(model.w_sum)
                print "bias of regression: ", sess.run(model.bias)

            #train
            batch = reader.next_batch()
            feed_dict = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, keep_prob, batch, 0.8, modality)
            sess.run(model.optimize, feed_dict=feed_dict)

            #print step
            if i%config.print_step == 0:
                print "train loss:",do_eval(sess, model.loss, feed_dict)
                print "validation loss: ", validation_last_loss

            #validation
            validation_loss = do_eval(sess, model.loss, validation_feed)

            #compare the validation with the last loss
            if validation_loss < validation_last_loss:
                validation_last_loss = validation_loss
                best_test_result = do_eval(sess, model.prediction, test_feed)

if __name__ == "__main__":
    tf.app.run()
