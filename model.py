# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import config as cfg

from uitls.tools import *

class lenet(object):
    """Lenet model"""
    def __init__(self):
        self.learning_rate = cfg.LEARNING_RATE
        self.keep_prod = cfg.KEEP_PROB
        self.num_classes = cfg.NUM_CLASSES

    def build(self, x):
        """build Lenet model"""
        conv1 = convLayer(x, 5, 5, 6, 1, 1, name='conv1', padding='VALID')
        pool1 = maxPoolLayer(conv1, 2, 2, 2, 2, name='pool1', padding='VALID')

        conv2 = convLayer(pool1, 5, 5, 16, 1, 1, name='conv2', padding='VALID')
        pool2 = maxPoolLayer(conv2, 2, 2, 2, 2, name='pool2', padding='VALID')
        pass