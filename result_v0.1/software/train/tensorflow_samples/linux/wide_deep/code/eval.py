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
    embeddings = tf.get_variable("emb", [FLAGS.dict_size, FLAGS.embedding_size], tf.float32,
                                 initializer=tf.random_uniform_initializer(-1.0, 1.0))

    def embedding_layer(input_):
        return tf.reduce_sum(tf.nn.embedding_lookup(embeddings, input_), axis=1)

    sparse_embed_seq = list(map(embedding_layer, words[1:-1]))
    concat = tf.concat(sparse_embed_seq + words[0:1], axis=1)
    label_y = words[-1]

    # wide_part
    fc_wide_w = tf.get_variable("fc_wide_w", [FLAGS.dense_nums, 1],
                                tf.float32,
                                initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(FLAGS.dense_nums))))
    fc_wide_b = tf.get_variable("fc_wide_b", [1], tf.float32,
                                initializer=tf.constant_initializer(value=0))

    # deep_part
    fc0_w = tf.get_variable("fc0_w", [FLAGS.embedding_size * FLAGS.slot_nums + FLAGS.dense_nums, 400],
                            tf.float32,
                            initializer=tf.random_normal_initializer(stddev=1.0/tf.sqrt(tf.to_float(FLAGS.embedding_size * FLAGS.slot_nums + FLAGS.dense_nums))))
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

    fc_wide_w = tf.reshape(fc_wide_w, (1, 13, 1))
    wide_predict = tf.matmul(words[0:1], fc_wide_w) + fc_wide_b
    wide_predict = tf.reshape(wide_predict, (-1, 1))
    fc0 = tf.nn.relu(tf.matmul(concat, fc0_w) + fc0_b)
    fc1 = tf.nn.relu(tf.matmul(fc0, fc1_w) + fc1_b)
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    deep_predict = tf.matmul(fc2, fc3_w) + fc3_b

    fc_predict = tf.add(wide_predict, deep_predict)
    predict = tf.nn.sigmoid(fc_predict)

    predict = tf.reshape(predict, [-1, 1])
    label_y = tf.reshape(label_y, [-1, 1])

    cost = tf.losses.log_loss(label_y, predict)
    avg_cost = tf.reduce_mean(cost)

    train_auc, train_update_op = tf.metrics.auc(
        labels=label_y,
        predictions=predict,
        name="auc")

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
