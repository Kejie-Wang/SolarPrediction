# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '21/11/2016'

import tensorflow as tf
import numpy as np
from config import Model_Config
from reader import Reader
from model import Model
from util import MSE_And_MAE, test_figure_plot

def main(_):
    #get the config
    config = Model_Config()

    n_step = config.n_step
    n_target = config.n_target
    n_input_ir = 36
    n_input_mete = 26

    width_image = config.width
    height_image = config.height

    epoch_size = config.epoch_size
    print_step = config.print_step

    test_num = config.test_num

    #define the input and output
    x_ir = tf.placeholder(tf.float32, [None, n_step, n_input_ir])
    x_mete = tf.placeholder(tf.float32, [None, n_step, n_input_mete])
    x_sky_cam = tf.placeholder(tf.float32, [None, n_step, height_image, width_image])
    y_ = tf.placeholder(tf.float32, [None, n_target])
    keep_prob = tf.placeholder(tf.float32)

    reader = Reader(config)

    model = Model([x_ir, x_mete, x_sky_cam], y_, keep_prob, config)

    prediction = model.prediction
    loss = model.loss
    optimize = model.optimize

    #new a saver to save the model
    saver = tf.train.Saver()

    validation_last_loss = float('inf')

    with tf.Session() as sess:
        # initialize all variables
        tf.global_variables_initializer().run()

        path = tf.train.latest_checkpoint('.')
        save_path = saver.restore(sess, path)

        for i in range(epoch_size):
            # test
            if i%config.test_step == 0:
                ir_test_input, mete_test_input, sky_cam_test_input, test_target = reader.get_test_set()
                test_feed = {x_ir:ir_test_input, x_mete:mete_test_input, x_sky_cam: sky_cam_test_input, keep_prob:1.0}
                test_result = sess.run(prediction, feed_dict=test_feed)

                #calculate the mse and mae
                mse, mae = MSE_And_MAE(test_target, test_result)
                print "Test MSE: ", mse
                print "Test MAE: ", mae

                validation_set = reader.get_validation_set()
                validation_feed = {x_ir:validation_set[0], x_mete:validation_set[1], x_sky_cam: validation_set[2], keep_prob:1.0}
                validation_result = sess.run(prediction, feed_dict=validation_feed)
                mse, mae = MSE_And_MAE(validation_set[3], validation_result)
                print "Validation MSE: ", mse
                print "Validation MAE: ", mae

                ir_train_input, mete_train_input, sky_cam_train_input, train_target = reader.next_batch()
                train_feed = {x_ir: ir_train_input, x_mete:mete_train_input, x_sky_cam:sky_cam_train_input, keep_prob:1.0}
                train_result = sess.run(prediction, feed_dict=train_feed)
                mse, mae = MSE_And_MAE(train_target, train_result)
                print "Train MSE: ", mse
                print "Train MAE: ", mae

                # test_figure_plot(test_target, test_result)

            #train
            batch = reader.next_batch()
            train_feed = {x_ir:batch[0], x_mete:batch[1], x_sky_cam: batch[2], y_:batch[3],keep_prob:0.5}
            sess.run(optimize, feed_dict=train_feed)

            #print step
            if i%config.print_step == 0:
                print "train loss:",sess.run(loss, feed_dict=train_feed)
                print "validation loss: ", validation_last_loss

            #validation
            validation_set = reader.get_validation_set()
            validation_feed = {x_ir:validation_set[0], x_mete:validation_set[1], x_sky_cam: validation_set[2], y_:validation_set[3],keep_prob:0.5}
            validation_loss = sess.run(loss,feed_dict=validation_feed)

            #compare the validation with the last loss
            if(validation_loss < validation_last_loss):
                validation_last_loss = validation_loss
            # else:
            #     # break
            #     print "break"

            # print "validation loss: ", validation_loss
            if i%100 == 0 and i > 0:
                save_path = saver.save(sess, "model.ckpt")
                print "save the model"

if __name__ == "__main__":
    tf.app.run()
