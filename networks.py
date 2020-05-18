import tensorflow as tf
from IPython import embed
import numpy as np
import tf_util


# def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
#     """ Feature Transform Net, input is BxNx1xK
#         Return:
#             Transformation matrix of size KxK """
#     batch_size = inputs.get_shape()[0].value
#     num_point = inputs.get_shape()[1].value
#
#     net = tf_util.conv2d(inputs, 64, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                          scope='tconv1', bn_decay=bn_decay)
#     net = tf_util.conv2d(net, 128, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                          scope='tconv2', bn_decay=bn_decay)
#     net = tf_util.conv2d(net, 1024, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                          scope='tconv3', bn_decay=bn_decay)
#     net = tf_util.max_pool2d(net, [num_point, 1],
#                              padding='VALID', scope='tmaxpool')
#
#     net = tf.reshape(net, [-1, 1024])
#     net = tf_util.fully_connected(net, 512, bn=tf.constant(True, dtype=tf.bool),
#                                   is_training=is_training,
#                                   scope='tfc1', bn_decay=bn_decay)
#     net = tf_util.fully_connected(net, 256, bn=tf.constant(True, dtype=tf.bool),
#                                   is_training=is_training,
#                                   scope='tfc2', bn_decay=bn_decay)
#
#     with tf.variable_scope('transform_feat') as sc:
#         weights = tf.get_variable('weights', [256, K * K],
#                                   initializer=tf.constant_initializer(0.0),
#                                   dtype=tf.float32)
#         biases = tf.get_variable('biases', [K * K],
#                                  initializer=tf.constant_initializer(0.0),
#                                  dtype=tf.float32)
#         biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
#         transform = tf.matmul(net, weights)
#         transform = tf.nn.bias_add(transform, biases)
#
#     transform = tf.reshape(transform, [-1, K, K])
#     return transform
#
#
# def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
#     """ Input (XYZ) Transform Net, input is BxNx3 gray image
#         Return:
#             Transformation matrix of size 3xK """
#     batch_size = point_cloud.get_shape()[0].value
#     num_point = point_cloud.get_shape()[1].value
#
#     input_image = tf.expand_dims(point_cloud, -1)
#     net = tf_util.conv2d(input_image, 64, [1, 3],
#                          padding='VALID', stride=[1, 1],
#                          bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                          scope='tconv1', bn_decay=bn_decay)
#     net = tf_util.conv2d(net, 128, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                          scope='tconv2', bn_decay=bn_decay)
#     net = tf_util.conv2d(net, 1024, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                          scope='tconv3', bn_decay=bn_decay)
#     net = tf_util.max_pool2d(net, [num_point, 1],
#                              padding='VALID', scope='tmaxpool')
#
#     net = tf.reshape(net, [batch_size, -1])
#     net = tf_util.fully_connected(net, 512, bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                                   scope='tfc1', bn_decay=bn_decay)
#     net = tf_util.fully_connected(net, 256, bn=tf.constant(True, dtype=tf.bool), is_training=is_training,
#                                   scope='tfc2', bn_decay=bn_decay)
#
#     with tf.variable_scope('transform_XYZ') as sc:
#         assert (K == 3)
#         weights = tf.get_variable('weights', [256, 3 * K],
#                                   initializer=tf.constant_initializer(0.0),
#                                   dtype=tf.float32)
#         biases = tf.get_variable('biases', [3 * K],
#                                  initializer=tf.constant_initializer(0.0),
#                                  dtype=tf.float32)
#         biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
#         transform = tf.matmul(net, weights)
#         transform = tf.nn.bias_add(transform, biases)
#
#     transform = tf.reshape(transform, [batch_size, 3, K])
#     return transform


def cloth_cnn_posh(opt, input_p, reuse=False):
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

        for filt in [64, 32, 32, 64]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))

        grasp = tf.layers.dense(net, 3, kernel_initializer=ni, activation=None)
        points = tf.layers.dense(net, opt.num * 3, kernel_initializer=ni, activation=None)

        grasp = tf.reshape(grasp, [-1, 3])
        points = tf.reshape(points, [-1, opt.num * 3])

        return grasp, points


def cloth_cnn_heatmap(opt, input_p, input_g, reuse=False):
    with tf.variable_scope('model_heatmap', reuse=reuse):
        ni = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        net = tf.concat([input_p, input_g], 1)
        for filt in [32, 64, 128, 256]:
            net = tf.nn.leaky_relu(tf.layers.dense(net, filt, kernel_initializer=ni))

        score = tf.layers.dense(net, 1, kernel_initializer=ni, activation=None)

        return score
