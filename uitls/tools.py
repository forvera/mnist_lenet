# -*- coding:utf-8 -*-
import tensorflow as tf
import config as cfg

def convLayer(x, kHeight, kWidth, featureNum, strideX, strideY, name, padding='SAME'):
    """conv layer"""
    channels = x.get_shape()[-1]
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weight', [kHeight, kWidth, channels, featureNum])
        biases = tf.get_variable('biases', [featureNum])
        conv = tf.nn.conv2d(x, weights, [1, strideX, strideY, 1], padding, name)
        conv_bias = tf.nn.bias_add(conv, biases)
    return tf.nn.relu(conv_bias)

def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding='SAME'):
    """max pooling"""
    return tf.nn.max_pool(x, [1, kHeight, kWidth, 1], [1, strideX, strideY, 1], padding, name)

def fcLayer(x, inputD, outputD, reluflag, name):
    """fully connect layer"""
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[inputD, outputD], dtype=tf.float32)
        biases = tf.get_variable('bias', shape=[outputD], dtype=tf.float32)
        x = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        if reluflag:
            return tf.nn.relu(x)
        return x

def dropout(x, keep_prob, name=None):
    """dropout"""
    return tf.nn.dropout(x, keep_prob, name)

def generateVariables():
    with tf.name_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal([5, 5, 1, 6], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[6], dtype=tf.float32), trainable=True, name='biases')
    with tf.name_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal([5, 5, 6, 16], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases')
    with tf.name_scope('conv3') as scope:
        weights = tf.Variable(tf.truncated_normal([5, 5, 16, 120], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[120], dtype=tf.float32), trainable=True, name='biases')
    with tf.name_scope('fc4') as scope:
        weights = tf.Variable(tf.truncated_normal([120, 84], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[84], dtype=tf.float32), trainable=True, name='biases')
    with tf.name_scope('fc5') as scope:
        weights = tf.Variable(tf.truncated_normal([84, cfg.NUM_CLASSES], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[cfg.NUM_CLASSES], dtype=tf.float32), trainable=True, name='biases')

def loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

def optimizer(loss_op, learning_rate):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

def accuracy(logits, y):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))
    return acc