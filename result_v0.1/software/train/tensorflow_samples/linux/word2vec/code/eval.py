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

import time
import argparse
import shutil
import sys
import os
import glob
import json
import random
from datetime import date, timedelta

import logging
import tensorflow as tf
import six
import re
import numpy as np

from net import infer_network
#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("embedding_size", 300, "Embedding size")

tf.app.flags.DEFINE_string(
    "test_data", 'test_data/questions-words.txt', "test data")
tf.app.flags.DEFINE_string("checkpoint_path", 'model',
                           "directory to save checkpoint file")
tf.app.flags.DEFINE_string(
    "dict_path", 'dict/word_id_dict.txt', "dict path")
tf.app.flags.DEFINE_string("task_mode", 'async', "task_mode")
tf.app.flags.DEFINE_string(
    "result_path", '', "directory to save evaluate result")

logging.basicConfig(
    format='AI-Rank-log - %(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tensorflow")
logger.setLevel(logging.INFO)


def BuildWord_IdMap(dict_path):
    word_to_id = dict()
    id_to_word = dict()
    with open(dict_path, 'r') as f:
        for line in f:
            word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
            id_to_word[int(line.split(' ')[1])] = line.split(' ')[0]
    return word_to_id, id_to_word


def native_to_unicode(s):
    if _is_unicode(s):
        return s
    try:
        return _to_unicode(s)
    except UnicodeDecodeError:
        res = _to_unicode(s, ignore_errors=True)
        return res


def _is_unicode(s):
    if six.PY2:
        if isinstance(s, unicode):
            return True
    else:
        if isinstance(s, str):
            return True
    return False


def _to_unicode(s, ignore_errors=False):
    if _is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def strip_lines(line, vocab):
    return _replace_oov(vocab, native_to_unicode(line))


def _replace_oov(original_vocab, line):
    """Replace out-of-vocab words with "<UNK>".
    This maintains compatibility with published results.
    Args:
        original_vocab: a set of strings (The standard vocabulary for the dataset)
        line: a unicode string - a space-delimited sequence of words.
    Returns:
        a unicode string - a space-delimited sequence of words.
    """
    return u" ".join([
        word if word in original_vocab else u"<UNK>" for word in line.split()
    ])


def read_analogies(word_to_id):
    questions = []
    questions_skipped = 0
    with open(FLAGS.test_data, "rb") as analogy_f:
        for line in analogy_f:
            if line.startswith(b":"):  # Skip comments.
                continue
            words = strip_lines(line.lower(), word_to_id)
            words = words.split()
            ids = [word_to_id[w] for w in words]
            questions.append(np.array(ids))
    logger.info("Eval analogy file: {}".format(FLAGS.test_data))
    logger.info("Questions: {}".format(len(questions)))
    return np.array(questions, dtype=np.int32)


def eval(sess, questions, analogy_pred_idx, analogy_a, analogy_b, analogy_c):
    start = 0
    total = questions.shape[0]
    correct = 0
    while start < total:
        limit = min(start + 2500, total)
        sub = questions[start:limit, :]
        idx = sess.run([analogy_pred_idx], {
            analogy_a: sub[:, 0],
            analogy_b: sub[:, 1],
            analogy_c: sub[:, 2]
        })[0]
        start = limit
        for question in xrange(sub.shape[0]):
            for j in xrange(4):
                if idx[question, j] == sub[question, 3]:
                    correct += 1
                    break
                elif idx[question, j] in sub[question, :3]:
                    continue
                else:
                    # The correct label is not the precision@1
                    break
    logger.info("Eval {} / {}, eval_accuracy = {}".format(correct, total,
                                                          correct * 100.0 / total))
    return correct * 100.0 / total


def filter_checkpoint(checkpoint_path):
    all_files = os.listdir(checkpoint_path)
    all_files = filter(lambda x: re.match(".*\.index$", x), all_files)
    all_files = [os.path.join(checkpoint_path, x[:-6]
                              ).decode('utf-8') for x in all_files]
    return all_files


def main(result_file_path, checkpoint_path):
    word_to_id, id_to_word = BuildWord_IdMap(FLAGS.dict_path)
    questions = read_analogies(word_to_id)
    vocab_size = len(id_to_word)
    pred_idx, analogy_a, analogy_b, analogy_c = infer_network(
        vocab_size, FLAGS.embedding_size)

    res = {}

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        all_models = filter_checkpoint(checkpoint_path)
        if ckpt and all_models:
            for path in all_models:
                global_step = path.split('/')[-1].split('-')[-1]

                logger.info("Start to inference ==> {}".format(path))
                saver.restore(sess, path)
                acc = eval(sess, questions, pred_idx,
                           analogy_a, analogy_b, analogy_c)
                res[global_step] = acc
                logger.info("Epoch: {}, eval_accuracy: {}".format(path, acc))
        else:
            logger.info('No checkpoint file found')


if __name__ == '__main__':
    logger.info("task_mode: %s" % FLAGS.task_mode)
    logger.info("checkpoint path: %s" % FLAGS.checkpoint_path)
    logger.info("result path is: %s" % FLAGS.result_path)
    logger.info("evaluate %s" % FLAGS.checkpoint_path)
    result_file_path = os.path.join(os.path.abspath(
        FLAGS.result_path), FLAGS.task_mode) + '.json'
    main(result_file_path, FLAGS.checkpoint_path)
