import tensorflow as tf
from IPython import embed
import numpy as np
import tf_util


def cnn_posh_pointnet(opt, input_p, is_training, bn_decay, reuse=False):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    input_p = tf.reshape(input_p, [-1, opt.num, 3])
    num_point = input_p.get_shape()[1].value
    input_image = tf.expand_dims(input_p, -1)

    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [1, 3],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    net = tf_util.max_pool2d(net, [num_point, 1],
                             padding='VALID', scope='maxpool')

    global_feat_expand = tf.tile(net, [1, num_point, 1, 1])

    net = tf_util.conv2d(global_feat_expand, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 3, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    net = tf.reshape(net, [-1, opt.num * 3])
    ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    net = tf.layers.dense(net, opt.num * 3, kernel_initializer=ni, activation=None)
    grasp = tf.layers.dense(tf.reshape(input_p, [-1, opt.num * 3]), 6, activation=None)
    return grasp, net


def cnn_posh(opt, input_p, reuse=False):
    with tf.variable_scope('model_posh', reuse=reuse):
        ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        net = input_p

        # num_point = net.get_shape()[1]
        # input_image = tf.expand_dims(net, -1)
        # net = tf_util.conv2d(input_image, 512, [1, 3],
        #                      padding='VALID', stride=[1, 1],
        #                      bn=tf.constant(True, dtype=tf.bool),
        #                      is_training=tf.constant(evals, dtype=tf.bool),
        #                      scope='conv1', bn_decay=bn_decay)
        # net = tf_util.conv2d(net, 512, [1, 1],
        #                      padding='VALID', stride=[1, 1],
        #                      bn=tf.constant(True, dtype=tf.bool),
        #                      is_training=tf.constant(evals, dtype=tf.bool),
        #                      scope='conv2', bn_decay=bn_decay)
        # # net = tf_util.max_pool2d(net, [num_point, 1],
        #                          padding='VALID', scope='maxpool')
        # net = tf.reshape(net, [-1, 512])
        # net = tf.layers.flatten(net)

        for filt in [512, 128, 32, 32, 128, 512]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))

        grasp = tf.layers.dense(net, 3, kernel_initializer=ni, activation=None)
        points = tf.layers.dense(net, opt.num * 3, kernel_initializer=ni, activation=None)

        grasp = tf.reshape(grasp, [-1, 6])
        points = tf.reshape(points, [-1, opt.num * 3])

        return grasp, points


def cnn_heatmap(opt, input_p, input_g, reuse=False):
    with tf.variable_scope('model_heatmap', reuse=reuse):
        ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        net = tf.concat([input_p, input_g], 1)
        for filt in [32, 64, 128, 256]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))

        score = tf.layers.dense(net, 1, kernel_initializer=ni, activation=None)

        return score
