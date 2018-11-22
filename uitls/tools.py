# -*- coding:utf-8 -*-
import tensorflow as tf

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