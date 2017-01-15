from __future__ import division
# -*- coding: utf-8 -*-
__author__= 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '05/01/2017'

import tensorflow as tf
import numpy as np
import os
import sys
from model import Model

HOUR_IN_A_DAY = 24

class Config:
    n_step = 24
    h_ahead = 9
    n_target = 1

    model = "lin"

    epsilon = 10
    C = 3

    quantile_rate = 0.3

    hidden_size = 500
    lr = 0.0005
    batch_size = 100
    epoch_size = 10000
    print_step = 200
    test_step = 500

ir_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/ir_train_data.csv"
mete_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv"
target_train_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv"

ir_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/ir_validation_data.csv"
mete_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/mete_validation_data.csv"
target_validation_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/validation/target_validation_data.csv"

ir_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/ir_test_data.csv"
mete_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv"
target_test_data_path = "../../dataset/NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv"

def get_valid_index(features, targets):
  num = len(features)
  missing_index = []
  for i in range(len(features)):
    if True in np.isnan(features[i]) or True in np.isnan(targets[i]):
      missing_index.append(i)
  print missing_index
  return np.setdiff1d(np.arange(num), np.array(missing_index))

def _make_saved_model_path(config):
    #make the saved model path
    saved_model_path = "../saved_model/"
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    saved_model_path += "same_day/"
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    saved_model_path += (model + "/")
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)
    if config.model == "lin" or config.model == "msvr":
        saved_model_path += (str(config.h_ahead) + "_ahead" + "__" + str(config.n_target) + "_output" + "__" + str(config.hidden_size) + "_hidden/")
    elif config.model == "quantile":
        saved_model_path += (str(config.h_ahead) + "_ahead" + "__" + str(config.n_target) + "_output" + "__" + \
                        str(config.epsilon) + "_epsilon" + "__" + str(config.quantile_rate) + "_rate" + "__" + str(config.hidden_size) + "_hidden/")
    if not os.path.exists(saved_model_path):
        os.mkdir(saved_model_path)

    return saved_model_path

def _make_output_path(config):
    output_path = "../output/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path += "same_day/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path += (config.model + "/")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path += str(config.h_ahead) + "_" + str(config.h_ahead + config.n_target) + ".csv"

    return output_path

config = Config()

#load data
# ir_train_raw_data = np.loadtxt(ir_train_data_path, delimiter=',')
# ir_validation_raw_data = np.loadtxt(ir_validation_data_path, delimiter=',')
# ir_test_raw_data = np.loadtxt(ir_test_data_path, delimiter=',')

mete_train_raw_data = np.loadtxt(mete_train_data_path, delimiter=',')
mete_validation_raw_data = np.loadtxt(mete_validation_data_path, delimiter=',')
mete_test_raw_data = np.loadtxt(mete_test_data_path, delimiter=',')

target_train_raw_data = np.loadtxt(target_train_data_path, delimiter=',')
target_validation_raw_data = np.loadtxt(target_validation_data_path, delimiter=',')
target_test_raw_data = np.loadtxt(target_test_data_path, delimiter=',')

np.set_printoptions(precision=4)
print np.array(sorted(target_test_raw_data[np.arange(config.h_ahead, len(target_test_raw_data), HOUR_IN_A_DAY)]))

n_input_mete = mete_train_raw_data.shape[1]
mete_train_data = np.reshape(mete_train_raw_data, [-1, HOUR_IN_A_DAY, n_input_mete])
target_train_data = np.reshape(target_train_raw_data, [-1, HOUR_IN_A_DAY])[:, config.h_ahead:config.h_ahead+config.n_target]
mete_validation_data = np.reshape(mete_validation_raw_data, [-1, HOUR_IN_A_DAY, n_input_mete])
target_validation_data = np.reshape(target_validation_raw_data, [-1, HOUR_IN_A_DAY])[:, config.h_ahead:config.h_ahead+config.n_target]
mete_test_data = np.reshape(mete_test_raw_data, [-1, HOUR_IN_A_DAY, n_input_mete])
target_test_data = np.reshape(target_test_raw_data, [-1, HOUR_IN_A_DAY])[:, config.h_ahead:config.h_ahead+config.n_target]

#get valid index
train_valid_index = get_valid_index(mete_train_data, target_train_data)
validation_valid_index = get_valid_index(mete_validation_data, target_validation_data)
test_valid_index = get_valid_index(mete_test_data, target_test_data)

#eliminate the missing value index
mete_train_data = mete_train_data[train_valid_index]
target_train_data = target_train_data[train_valid_index]
mete_validation_data = mete_validation_data[validation_valid_index]
target_validation_data = target_validation_data[validation_valid_index]
mete_test_data = mete_test_data[test_valid_index]
target_test_data = target_test_data[test_valid_index]

#concatenate all valid data
mete_valid_data = np.concatenate((mete_train_data, mete_validation_data, mete_test_data), axis=0)

#feature scale
mete_mean = np.mean(mete_valid_data, axis=0)
mete_std = np.std(mete_valid_data, axis=0)
mete_std[mete_std==0] = 1.0
mete_train_data = (mete_train_data - mete_mean) / mete_std
mete_validation_data = (mete_validation_data - mete_mean) / mete_std
mete_test_data = (mete_test_data - mete_mean) / mete_std

train_num = len(mete_train_data)
validation_num =len(mete_validation_data)
test_num = len(mete_test_data)

print "----------------------Running model info----------------------"
print "valid train number:", train_num
print "valid validation number:", validation_num
print "valid test number:", test_num
print "-------------------------------------------------------------"

model_num = 1

x_mete = tf.placeholder(tf.float32, [None, config.n_step, n_input_mete])
y_ = tf.placeholder(tf.float32, [None, config.n_target])
keep_prob = tf.placeholder(tf.float32)

models = []
predictions = []
losses = []

for i in range(model_num):
    config.h_ahead += 1
    models.append(Model([x_mete, y_], config, keep_prob, i))

#new a saver to save the model
# saver = tf.train.Saver()
validation_last_loss = ['inf'] * model_num

with tf.Session() as sess:
  # initialize all variables
  # the new method in r0.12
  # if you are use the earlier version, please replace it with initial_all_variable
  tf.global_variables_initializer().run()

  # path = tf.train.latest_checkpoint(saved_model_path)
  # if not (path is None):
  #     save_path = saver.restore(sess, path)
  #     print "restore model"

  for i in range(config.epoch_size):
    index = np.random.choice(np.arange(train_num), config.batch_size, replace=False)
    mete_input, target = mete_train_data[index], target_train_data[index]
    for j in range(model_num):
        sess.run(models[j].optimize, feed_dict={x_mete: mete_input, y_: target, keep_prob: 0.8})

    if i%25 == 0:
        for j in range(model_num):
            sess.run(models[j].optimize, feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 0.8})

    if i%config.print_step == 0:
        for j in range(model_num):
            l = sess.run(models[j].loss, feed_dict={x_mete: mete_train_data, y_: target_train_data, keep_prob: 1.0})
            print "Model", j, "Step %d train loss: %.4f" %(i, l), "sum of weight", sess.run(models[j].w_sum), "bias", sess.run(models[j].bias)
            # print "Model", j, "sum of weight", sess.run(models[j].w_sum)

    if i%config.test_step == 0:
      print "-----------Do a test----------"
      for j in range(model_num):
          print "----model", j
          if config.model == "msvr" or config.model == "lin":
              rmse, mae = sess.run([models[j].rmse, models[j].mae], feed_dict={x_mete: mete_train_data, y_: target_train_data, keep_prob: 1.0})
              print "\ttrain rmse:",rmse
              print "\ttrain mae:", mae

              l = sess.run(models[j].loss, feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
              print "\tvalidation loss: %.4f" %(l)
              rmse, mae = sess.run([models[j].rmse, models[j].mae], feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
              print "\tvalidation rmse:",rmse
              print "\tvalidation mae:", mae

              l = sess.run(models[j].loss, feed_dict={x_mete: mete_test_data, y_: target_test_data, keep_prob: 1.0})
              print "\ttest loss: %.4f" %(l)
              rmse, mae = sess.run([models[j].rmse, models[j].mae], feed_dict={x_mete: mete_test_data, y_: target_test_data, keep_prob: 1.0})
              print "\ttest rmse:",rmse
              print "\ttest mae:", mae
          elif config.model == "quantile":
              rate = sess.run(models[j].coverage_rate, feed_dict={x_mete: mete_train_data, y_: target_train_data, keep_prob: 1.0})
              print "\ttrain rate", rate
              rate = sess.run(models[j].coverage_rate, feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
              print "\tvalidation rate:", rate
              rate = sess.run(models[j].coverage_rate, feed_dict={x_mete: mete_test_data, y_: target_test_data, keep_prob: 1.0})
              print "\ttest rate", rate
      print "-----------finish the test----------"

      if i%100 == 0 and i>0:
          for j in range(model_num):
              l = sess.run(models[j].loss, feed_dict={x_mete: mete_validation_data, y_: target_validation_data, keep_prob: 1.0})
              if l < validation_last_loss[j]:
                  validation_last_loss[j] = l
            #   pred = sess.run(prediction, feed_dict={x_mete: mete_test_data, keep_prob: 1.0})
            #   np.savetxt(output_path, pred, fmt="%.4f", delimiter=',')
            #   print "save the output"
            #   save_path = saver.save(sess, saved_model_path + "model.ckpt")
            #   print "save the model to ", save_path
