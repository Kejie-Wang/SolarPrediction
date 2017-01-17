# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import tensorflow as tf
import numpy as np
from config import Model_Config
from reader import Reader
from model import Model
from util import MSE_And_MAE
import os

def fill_feed_dict(x_ir_placeholder, x_mete_placeholder, x_sky_cam_placeholder, y_, feed_data, modality):
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
    y_ = tf.placeholder(tf.float32, [None, n_target])

    reader = Reader(config)

    model = Model([x_ir, x_mete, x_sky_cam], y_, config)

    #new a saver to save the model
    saver = tf.train.Saver()

    validation_last_loss = float('inf')
    best_test_result = None

    save_folder_path = "./" + str(config.h_ahead) + "-" + str(config.h_ahead+config.n_target) + "/"
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    print save_folder_path

    with tf.Session() as sess:
        # initialize all variables
        tf.global_variables_initializer().run()

        path = tf.train.latest_checkpoint(save_folder_path)
        if not (path is None):
            save_path = saver.restore(sess, path)
            print "restore model"

        train_set = reader.get_train_set()
        validation_set = reader.get_validation_set()
        test_set = reader.get_test_set()

        train_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, train_set, modality)
        validation_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, validation_set, modality)
        test_feed = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, test_set, modality)

        for i in range(epoch_size):
            # test
            if i%config.test_step == 0:
                #calculate the mse and mae
                rmse, mae = do_eval(sess, [model.rmse, model.mae], test_feed)
                print "Test RMSE: ", rmse, "Test MAE: ", mae

                rmse, mae = do_eval(sess, [model.rmse, model.mae], validation_feed)
                print "Validation RMSE: ", rmse, "Validation MAE: ", mae

                rmse, mae = do_eval(sess, [model.rmse, model.mae], train_feed)
                print "Train RMSE: ", rmse, "Train MAE: ", mae

                print "sum of w: ", sess.run(model.w_sum)

                print "\n"
                print "bias of regression: ", sess.run(model.bias)

                # test_result_path = "../output/" + str(config.regressor) + "/" + str(config.h_ahead) + "_" + str(config.h_ahead + config.n_target) + ".res"
                # np.savetxt(test_result_path, best_test_result, fmt="%.4f", delimiter=',')

            #train
            batch = reader.next_batch()
            feed_dict = fill_feed_dict(x_ir, x_mete, x_sky_cam, y_, batch, modality)
            sess.run(model.optimize, feed_dict=train_feed)

            #print step
            if i%config.print_step == 0:
                print "train loss:",do_eval(sess, model.loss, feed_dict)
                print "validation loss: ", validation_last_loss

            #validation
            validation_loss = do_eval(sess, model.loss, validation_feed)

            if i%50 == 0 and i > 0 and validation_loss < validation_last_loss:
                save_path = saver.save(sess, save_folder_path + "model.ckpt")
                print "save the model to ", save_path

            #compare the validation with the last loss
            if validation_loss < validation_last_loss:
                validation_last_loss = validation_loss
                best_test_result = do_eval(sess, model.prediction, test_feed)

            # print "validation loss: ", validation_loss
            if i%100 == 0 and i > 0:
                save_path = saver.save(sess, "model.ckpt")
                print "save the model"


        # test_result_path = "../output/" + str(config.regressor) + "/" + str(config.h_ahead) + "_" + str(config.h_ahead + config.n_target) + ".res"
        # np.savetxt(test_result_path, best_test_result, fmt="%.4f", delimiter=',')

if __name__ == "__main__":
    tf.app.run()
