# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import config as cfg
from uitls.tools import generateVariables, loss, optimizer, accuracy
from model import lenet
from tensorflow.examples.tutorials.mnist import input_data

def train():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    batch_size = cfg.BATCH_SIZE
    learning_rate = cfg.LEARNING_RATE
    parameter_path = cfg.PARAMETER_FILE
    num_epoches = cfg.MAX_ITER
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y = tf.placeholder(tf.float32, [None, cfg.NUM_CLASSES])
    keep_prob = tf.placeholder(tf.float32)

    generateVariables()
    model = lenet()
    logits = model.build(x, keep_prob)
    loss_op = loss(logits, y)
    optimizer_op = optimizer(loss_op, cfg.LEARNING_RATE)

    acc = accuracy(logits, y)

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epoches):
                batch = mnist.train.next_batch(batch_size)
                batch_x = tf.reshape(batch[0], [-1, 28, 28, 1])
                batch_y = tf.reshape(batch[1], [-1, cfg.NUM_CLASSES])
                if (epoch+1) % 100 == 0:
                    sess.run(acc, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (epoch+1, acc))
                sess.run(optimizer_op, feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.KEEP_PROB})
    pass

if __name__ == '__main__':
    train()