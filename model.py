# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import config as cfg

from uitls.tools import *

class lenet(object):
    """Lenet model"""
    def __init__(self):
        self.num_classes = cfg.NUM_CLASSES

    def build(self, x, keep_prob):
        """build Lenet model"""
        conv1 = convLayer(x, 5, 5, 6, 1, 1, name='conv1', padding='SAME')
        pool1 = maxPoolLayer(conv1, 2, 2, 2, 2, name='pool1', padding='VALID')

        conv2 = convLayer(pool1, 5, 5, 16, 1, 1, name='conv2', padding='VALID')
        pool2 = maxPoolLayer(conv2, 2, 2, 2, 2, name='pool2', padding='VALID')

        conv3 = convLayer(pool2, 5, 5, 120, 1, 1, name='conv3', padding='VALID')

        flattened_shape = np.prod([s.value for s in conv3.get_shape()[1:]])
        flatten = tf.reshape(conv3, [-1, flattened_shape])

        net = fcLayer(flatten, flatten.get_shape()[-1], 84, reluflag=True, name='fc4')
        net = dropout(net, keep_prob, name='dropout')
        net = fcLayer(net, 84, self.num_classes, reluflag=False, name='fc5')
        return net