import tensorflow as tf
from IPython import embed
import numpy as np
import tf_util


def cnn_posh_pointnet(opt, input_p, is_training, bn_decay, reuse=False):
    ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope('model_posh_pts', reuse=reuse):
        input_p_ = tf.reshape(input_p, [-1, opt.num, 3])
        num_point = input_p_.get_shape()[1].value
        input_image = tf.expand_dims(input_p_, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(input_image, 32, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf.reduce_sum(net, axis=1, keep_dims=True)
        # net = tf_util.avg_pool2d(net, [num_point, 1],
        #                          padding='VALID', scope='avgpool')
        # net = tf.tile(net, [1, num_point, 1, 1])
        # net = tf.squeeze(net, [2])  # BxNxC
        net = tf.reshape(net, [-1, 64])
        for filt in [4096, 4096]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))
        net = tf.layers.dense(net, opt.num * 3, kernel_initializer=ni, activation=None)

    with tf.variable_scope('model_posh_grasp', reuse=reuse):
        x = input_p
        for filt in [512, 128, 32, 32, 128, 512]:
            x = tf.nn.leaky_relu(tf.layers.dense(x, filt, kernel_initializer=ni))
        grasp = tf.layers.dense(x, 6, kernel_initializer=ni, activation=None)

    return grasp, net


def cnn_posh(opt, input_p, reuse=False):
    with tf.variable_scope('model_posh', reuse=reuse):
        ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        net = input_p

        for filt in [4096, 4096]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))

        grasp = tf.layers.dense(net, 6, kernel_initializer=ni, activation=None)
        points = tf.layers.dense(net, opt.num * 3, kernel_initializer=ni, activation=None)

        grasp = tf.reshape(grasp, [-1, 6])
        points = tf.reshape(points, [-1, opt.num * 3])

        return grasp, points


def cnn_heatmap(opt, input_p, input_g, is_training, reuse=False):
    with tf.variable_scope('model_heatmap', reuse=reuse):
        ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        net = tf.concat([input_p, input_g], 1)

        net = tf.layers.dropout(net, 0.2, training=is_training)
        for filt in [32, 64, 128, 256]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))

        score = tf.layers.dense(net, 1, kernel_initializer=ni, activation=None)

        return score
