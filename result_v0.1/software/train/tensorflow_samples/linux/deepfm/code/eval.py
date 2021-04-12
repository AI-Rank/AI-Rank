#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import argparse
import shutil
import sys
import os
import glob
import json
import random
import logging
from datetime import date, timedelta
import math
import tensorflow as tf
import re
import data_generator as reader

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 10, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 20, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1000, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
tf.app.flags.DEFINE_integer("dict_size", 1000001, "dict_size")
tf.app.flags.DEFINE_integer("dense_nums", 13, "dense feature num")
tf.app.flags.DEFINE_integer("slot_nums", 26, "sparse feature num")

tf.app.flags.DEFINE_string("test_data_dir", 'test_data', "test data dir")
tf.app.flags.DEFINE_string("task_mode", '', "task_mode")
tf.app.flags.DEFINE_string("checkpoint_path", '',
                           "directory to save checkpoint file")

logging.basicConfig(
    format='AI-Rank-log - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tensorflow")
logger.setLevel(logging.INFO)


def get_file_list():
    data_dir = FLAGS.test_data_dir
    data_files = os.listdir(data_dir)
    file_list = list()
    for data_file in data_files:
        file_list.append(data_dir + '/' + data_file)
    logger.info("File list:" + str(file_list))
    return file_list


def get_batch(reader, batch_size):
    example_batch = []
    for _ in range(FLAGS.slot_nums + 2):
        example_batch.append([])
    for example in reader():
        for i in range(len(example)):
            example_batch[i].append(example[i])
        if len(example_batch[0]) >= batch_size:
            yield example_batch
            for _ in range(FLAGS.slot_nums + 2):
                example_batch[_] = []


def model(words):
    test_sparse_input = words[1:-1]
    test_dense_input = words[0:1]
    test_label_input = words[-1]

    init_value = 0.1
    sparse_word = []
    sparse_one_word = []
    dense_word = []

    embedding_one = tf.get_variable("emb_one", [FLAGS.dict_size, 1], tf.float32,
                                    initializer=tf.truncated_normal_initializer(
                                        mean=0.0,
                                        stddev=init_value / math.sqrt(float(FLAGS.embedding_size))))
    embeddings = tf.get_variable("emb", [FLAGS.dict_size, FLAGS.embedding_size], tf.float32,
                                 initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                             stddev=init_value / math.sqrt(float(FLAGS.embedding_size))))

    for i in range(26):
        key = 'C' + str(i)
        sparse_word.append(tf.nn.embedding_lookup(
            embeddings, test_sparse_input[i]))
        sparse_one_word.append(tf.nn.embedding_lookup(
            embedding_one, test_sparse_input[i]))

    dense_concat = test_dense_input[0]  # tf.concat(dense_word, axis=1)

    #### FM ####
    dense_w_one = tf.get_variable(
        "dense_w_one", [13], tf.float32, initializer=tf.constant_initializer(value=1.0))
    dense_w = tf.get_variable(
        "dense_w", [1, 13, 10], tf.float32, initializer=tf.constant_initializer(value=1.0))

    # -------------------- first order term  --------------------
    sparse_embedding_one = tf.concat(sparse_one_word, axis=1)
    sparse_embedding_one = tf.reshape(sparse_embedding_one, [-1, 26, 1])

    dense_emb_one = tf.multiply(dense_concat, dense_w_one)
    dense_emb_one = tf.expand_dims(dense_emb_one, 2)

    y_first_order = tf.reduce_sum(sparse_embedding_one, 1) + \
        tf.reduce_sum(dense_emb_one, 1)

    # -------------------- second order term  --------------------
    sparse_embedding = tf.concat(sparse_word, axis=1)
    sparse_embedding = tf.reshape(sparse_embedding, (-1, 26, 10))
    dense_input_re = tf.expand_dims(dense_concat, axis=2)
    dense_embedding = tf.multiply(dense_input_re, dense_w)
    feat_embeddings = tf.concat(
        [sparse_embedding, dense_embedding], axis=1)

    # sum_square part
    summed_features_emb = tf.reduce_sum(feat_embeddings, 1)
    summed_features_square = tf.square(summed_features_emb)

    # square_sum part
    squared_feature_emb = tf.square(feat_embeddings)
    squared_sum_features_emb = tf.reduce_sum(squared_feature_emb, 1)

    y_second_order = 0.5 * \
        tf.reduce_sum(summed_features_square - squared_sum_features_emb, 1)
    y_second_order = tf.reshape(y_second_order, (-1, 1))

    #### Deep ####
    fc0_w = tf.get_variable("fc0_w", [FLAGS.embedding_size * (FLAGS.slot_nums + 13), 400],
                            tf.float32,
                            initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(FLAGS.embedding_size * (FLAGS.slot_nums + FLAGS.dense_nums)))))
    fc0_b = tf.get_variable("fc0_b", [400], tf.float32,
                            initializer=tf.constant_initializer(value=0))
    fc1_w = tf.get_variable("fc1_w", [400, 400], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(400))))
    fc1_b = tf.get_variable("fc1_b", [400], tf.float32,
                            initializer=tf.constant_initializer(value=0))
    fc2_w = tf.get_variable("fc2_w", [400, 400], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(400))))
    fc2_b = tf.get_variable("fc2_b", [400], tf.float32,
                            initializer=tf.constant_initializer(value=0))
    fc3_w = tf.get_variable("fc3_w", [400, 1], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(400))))
    fc3_b = tf.get_variable("fc3_b", [1], tf.float32,
                            initializer=tf.constant_initializer(value=0))

    feat_embeddings_re = tf.reshape(feat_embeddings, (-1, 390))
    fc0 = tf.nn.relu(tf.matmul(feat_embeddings_re, fc0_w) + fc0_b)
    fc1 = tf.nn.relu(tf.matmul(fc0, fc1_w) + fc1_b)
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    deep_predict = tf.matmul(fc2, fc3_w) + fc3_b

    fc_predict = y_first_order + y_second_order + deep_predict
    predict = tf.nn.sigmoid(fc_predict)

    predict = tf.reshape(predict, [-1, 1])
    label_y = tf.reshape(test_label_input, [-1, 1])

    train_auc, train_update_op = tf.metrics.auc(
        labels=label_y,
        predictions=predict, name='auc')

    cost = tf.losses.log_loss(label_y, predict)
    avg_cost = tf.reduce_mean(cost)

    return train_auc, train_update_op, avg_cost


def main(checkpoint_path):
    result = {}
    file_list = get_file_list()
    logger.info("there are a total of %d test files" % (len(file_list)))
    logger.info(file_list)
    test_generator = reader.CriteoDataset(FLAGS.dict_size)

    dense_input = tf.placeholder(
        tf.float32, [None, FLAGS.dense_nums], name="dense_input")
    sparse_input = [tf.placeholder(
        tf.int64, [None, 1], name="C" + str(i)) for i in range(1, 27)]
    label_y = tf.placeholder(tf.int64, [None, 1], name="label")
    words = [dense_input] + sparse_input + [label_y]

    auc, update_op, avg_cost = model(words)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.all_model_checkpoint_paths:
            for path in ckpt.all_model_checkpoint_paths:
                sess.run(tf.local_variables_initializer())
                global_step = path.split('/')[-1].split('-')[-1]
                logger.info("Start to inference ==> %s" % (path))
                saver.restore(sess, path)
                local_step = 0
                auc_val = 0.0
                for words_input in get_batch(test_generator.test(file_list), FLAGS.batch_size):
                    feed_dict = {}
                    for i, item in enumerate(words):
                        feed_dict[item] = words_input[i]
                    auc_val, _, avg_cost_val = sess.run(
                        [auc, update_op, avg_cost], feed_dict=feed_dict)
                    if local_step % 100 == 0:
                        logger.info("global step: %s, eavl batch step: %d, eval_auc: %f, eval_loss: %f" % (
                            global_step, local_step, auc_val, avg_cost_val))
                    local_step += 1
                result[global_step] = str(auc_val)

        else:
            logger.info('No checkpoint file found')


if __name__ == '__main__':
    logger.info("task_mode: %s" % FLAGS.task_mode)
    logger.info("checkpoint path: %s" % FLAGS.checkpoint_path)
    logger.info("evaluate %s" % FLAGS.checkpoint_path)
    main(FLAGS.checkpoint_path)
