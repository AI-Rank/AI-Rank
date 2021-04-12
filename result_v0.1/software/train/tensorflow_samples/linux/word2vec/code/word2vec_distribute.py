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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
import reader
from net import skip_gram_word2vec
import logging
flags = tf.app.flags

flags.DEFINE_string("model_output_dir", 'model', "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_string("train_data_dir", 'train_data',
                    "Training text data directory.")
flags.DEFINE_integer("embedding_size", 300, "The embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 5, "Number of epochs to train."
                     "Each epoch processes the training data once completely.")
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 5,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 100,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("num_threads", 16, "num threads")

flags.DEFINE_string("dict_path", 'dict/word_count_dict.txt', "dict path")
flags.DEFINE_integer("save_steps", 30000000,
                     "The number of step to save (default: 30000000)")
flags.DEFINE_string("dist_mode", "sync", "sync_mode or async_mode")
flags.DEFINE_integer("is_local", 1, "local or mpi cluster")
FLAGS = flags.FLAGS

logging.basicConfig(
    format='AI-Rank-log - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tensorflow")
logger.setLevel(logging.INFO)


def get_batch(reader, batch_size):
    example_batch = []
    label_batch = []
    for example, label in reader():
        example_batch.append(example)
        label_batch.append(label)
        if len(example_batch) >= batch_size:
            yield example_batch, label_batch
            example_batch = []
            label_batch = []


def GetFileList(data_dir, trainer_nums, trainer_id):
    data_files = os.listdir(data_dir)
    if not FLAGS.is_local:
        return data_files

    total_files = len(data_files)
    remainder = int(total_files % trainer_nums)
    blocksize = int(total_files / trainer_nums)

    blocks = [blocksize] * trainer_nums
    for i in range(remainder):
        blocks[i] += 1
    trainer_files = [[]] * trainer_nums
    begin = 0
    for i in range(trainer_nums):
        trainer_files[i] = data_files[begin:begin + blocks[i]]
        begin += blocks[i]

    return trainer_files[trainer_id]


def get_example_num(file_list):
    file_list = [FLAGS.train_data_dir + '/' +
                 data_file for data_file in file_list]
    count = 0
    for f in file_list:
        last_count = count
        for index, line in enumerate(open(f, 'r')):
            line = line.rstrip().split()
            count += len(line)
        logger.info("load_data, file: %s has %s words" % (f, count-last_count))
    logger.info("Total words: %s" % count)
    return count


def main(_):
    ps_hosts = os.getenv("PADDLE_PSERVERS_IP_PORT_LIST").split(",")
    worker_hosts = os.getenv("PADDLE_WORKERS_IP_PORT_LIST").split(",")
    role = os.getenv("TRAINING_ROLE")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts,
                                    "worker": worker_hosts})

    if role == "PSERVER":
        pserver_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=pserver_id)
        server.join()
    elif role == "TRAINER":
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        server = tf.train.Server(cluster, job_name="worker",
                                 task_index=trainer_id)
        is_chief = (trainer_id == 0)
        num_workers = len(worker_hosts)
        device_setter = tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % trainer_id,
            cluster=cluster)
        with tf.device(device_setter):
            global_step = tf.Variable(0, name="global_step")

            filelist = GetFileList(FLAGS.train_data_dir,
                                   num_workers, trainer_id)
            all_examples = get_example_num(filelist)
            logger.info("train_file_list: %s" % str(filelist))
            logger.info("there are a total of %d files, %d words" %
                        (len(filelist), all_examples))
            word2vec_reader = reader.Word2VecReader(FLAGS.dict_path, FLAGS.train_data_dir,
                                                    filelist, 0, 1)
            logger.info("dict_size: {}".format(word2vec_reader.dict_size))

            examples, labels, loss = skip_gram_word2vec(word2vec_reader.dict_size, FLAGS.embedding_size, FLAGS.batch_size, np.array(
                word2vec_reader.id_frequencys), FLAGS.num_neg_samples)
            lr = tf.train.exponential_decay(
                learning_rate=FLAGS.learning_rate, global_step=global_step, decay_steps=100000, decay_rate=0.999, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
            hooks = []
            if FLAGS.dist_mode == "sync":
                optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                                                           replicas_to_aggregate=num_workers,
                                                           total_num_replicas=num_workers)
                hooks.append(optimizer.make_session_run_hook(is_chief))
            saver = tf.train.Saver(max_to_keep=10)
            saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.model_output_dir,
                                                      save_steps=FLAGS.save_steps,
                                                      saver=saver)
            hooks.append(saver_hook)
            train_op = optimizer.minimize(loss, global_step=global_step)
            sess_config = tf.ConfigProto(allow_soft_placement=True,
                                         log_device_placement=False,
                                         inter_op_parallelism_threads=FLAGS.num_threads,
                                         intra_op_parallelism_threads=FLAGS.num_threads)

            logger.info("test_begin")
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=is_chief,
                                                   hooks=hooks,
                                                   config=sess_config) as session:

                for epoch in xrange(FLAGS.epochs_to_train):
                    start_time = time.time()
                    batch_id = 0
                    for examples_, labels_, in get_batch(word2vec_reader.train(), FLAGS.batch_size):
                        feed_dict = {}
                        feed_dict[examples] = examples_
                        feed_dict[labels] = labels_
                        _, loss_, step_ = session.run(
                            [train_op, loss, global_step], feed_dict=feed_dict)
                        if batch_id % 1000 == 0:
                            logger.info("total_epoch_cnt %4d, Step %8d, local step %8d, loss = %6.2f" % (
                                epoch, step_, batch_id, loss_))
                        batch_id += 1
                    now = time.time()
                    speed = float(all_examples) / float(now - start_time)
                    logger.info(
                        "total_epoch_cnt: {} total time: {} avg_ips: {} word/s".format(epoch, now - start_time, speed))

            logger.info("test_finish")


if __name__ == "__main__":
    tf.app.run()
