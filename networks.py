import tensorflow as tf
from IPython import embed
import numpy as np
import tf_util


def cnn_posh_pointnet(opt, input_p, is_training, bn_decay, reuse=False):
    ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope('model_posh_pts', reuse=reuse):
        input_p = tf.reshape(input_p, [-1, opt.num, 3])
        num_point = input_p.get_shape()[1].value
        x = tf.expand_dims(input_p, -1)

        # Point functions (MLP implemented as conv2d)
        net = tf_util.conv2d(x, 32, [1, 3],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 256, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=is_training,
                             scope='conv4', bn_decay=bn_decay)

        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')

        for filt in [128, 128, 64, 64]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))
            net = tf.contrib.layers.batch_norm(net, decay=0.99, center=True, scale=True, reuse=False)

        net = tf.reshape(net, [-1, 4, 4, 4, 1])
        for i, filt in enumerate([64, 64]):
            net = tf.nn.leaky_relu(tf.layers.conv3d_transpose(net, filt, 2, 2, use_bias=False))
            net = tf.contrib.layers.batch_norm(net, decay=0.99, center=True, scale=True, reuse=False)
        net = tf.layers.conv3d_transpose(net, 1, 2, 2, use_bias=False)
        points = tf.reshape(net, [-1, 32, 32, 32])

    with tf.variable_scope('model_posh_grasp', reuse=reuse):
        x = tf.reshape(input_p, [-1, opt.num * 3])
        for filt in [64, 64]:
            x = tf.nn.leaky_relu(tf.layers.dense(x, filt, kernel_initializer=ni))
        grasp = tf.layers.dense(x, 6, kernel_initializer=ni, activation=None)

    return grasp, points, tf.nn.sigmoid(points)


def cnn_posh(opt, input_p, reuse=False):
    with tf.variable_scope('model_posh', reuse=reuse):
        ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        net_ = input_p
        for filt in [512, 256, 128, 64]:
            net_ = tf.nn.leaky_relu(tf.layers.dense(net_, filt, kernel_initializer=ni))
            net_ = tf.contrib.layers.batch_norm(net_, decay=0.99, center=True, scale=True, reuse=False)
        net = tf.layers.dense(net_, 64, kernel_initializer=ni)
        net = tf.reshape(net, [-1, 4, 4, 4, 1])
        net = tf.nn.leaky_relu(tf.layers.conv3d_transpose(net, 64, 2, 2, use_bias=False))
        net = tf.layers.conv3d_transpose(net, 1, 4, 4, use_bias=False)
        points = tf.reshape(net, [-1, 32, 32, 32])

        grasp = tf.layers.dense(net_, 6, kernel_initializer=ni, activation=None)
        # points = tf.layers.dense(net, opt.num * 3, kernel_initializer=ni, activation=None)

        grasp = tf.reshape(grasp, [-1, 6])
        # points = tf.reshape(points, [-1, opt.num * 3])

        return grasp, points, tf.nn.sigmoid(points)


def cnn_heatmap(opt, input_p, input_g, is_training, reuse=False):
    with tf.variable_scope('model_heatmap', reuse=reuse):
        ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        net = tf.concat([input_p, input_g], 1)

        net = tf.layers.dropout(net, 0.2, training=is_training)
        for filt in [32, 64, 128, 256]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))

        score = tf.layers.dense(net, 1, kernel_initializer=ni, activation=None)

        return score
