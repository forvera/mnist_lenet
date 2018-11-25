# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import datetime
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

    # Summaries
    tf.summary.scalar('train_loss', loss_op)
    tf.summary.scalar('train_accuracy', acc)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logs/train/')
    val_writer = tf.summary.FileWriter('logs/val/')
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(mnist.train.labels.size / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(mnist.test.labels.size / batch_size).astype(np.int16) 

    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            sess.run(tf.global_variables_initializer())
            train_writer.add_graph(sess.graph)
            for epoch in range(num_epoches):
                print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
                step = 1
                while step < train_batches_per_epoch:
                    batch = mnist.train.next_batch(batch_size)
                    batch_x = np.reshape(batch[0], [-1, 28, 28, 1])
                    batch_y = np.reshape(batch[1], [-1, cfg.NUM_CLASSES])
                    sess.run(optimizer_op, feed_dict={x: batch_x, y: batch_y, keep_prob: cfg.KEEP_PROB})
                    
                    # Logging
                    if (step+1) % 100 == 0:
                        train_acc = sess.run(acc, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                        print('step %d, training accuracy %g' % (step+1, train_acc))
                        s = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                        train_writer.add_summary(s, epoch*train_batches_per_epoch + step)
                    step += 1
                
                # Epoch completed, start validation
                print("{} Start validation".format(datetime.datetime.now()))
                test_acc = 0.
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    batch = mnist.test.next_batch(batch_size)
                    batch_x = np.reshape(batch[0], [-1, 28, 28, 1])
                    batch_y = np.reshape(batch[1], [-1, cfg.NUM_CLASSES])
                    val_acc = sess.run(acc, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    test_acc += val_acc
                    test_count += 1
                test_acc /= test_count
                s = tf.Summary(value=[tf.Summary.Value(tag='validation_accuracy', simple_value=test_acc)])
                val_writer.add_summary(s, epoch+1)
                print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

                checkpoint_path = os.path.join('checkpoint/', 'model_epoch'+str(epoch+1)+'.ckpt')
                save_path = saver.save(sess, checkpoint_path)
                print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))


if __name__ == '__main__':
    train()