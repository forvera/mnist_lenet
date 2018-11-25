# -*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from model import lenet
import config as cfg
import matplotlib.pyplot as plt

class inference:
    def __init__(self):
        self.lenet = lenet()
        self.sess = tf.Session()
        self.parameter_path = cfg.PARAMETER_FILE

    def predict(self, image):
        img = image.convert('L')
        img = img.resize([28, 28], Image.ANTIALIAS)
        image_input = np.array(img, dtype=np.float32) / 255
        # plt.imshow(image_input)
        # plt.show()
        image_input = np.reshape(image_input, [1, 28, 28, 1])
        image_input = tf.convert_to_tensor(image_input)
        predict_op = self.lenet.build(image_input, 1.0)
        predict = tf.argmax(tf.nn.softmax(predict_op), 1)
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.parameter_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        prediction = self.sess.run(predict)
        return prediction