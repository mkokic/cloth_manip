import tensorflow as tf
import os


def counter(scope='counter'):
    with tf.variable_scope(scope):
        counter = tf.Variable(0, dtype=tf.int32, name='counter')
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt


def load_checkpoint(ckpt_dir_or_file, session, var_list=None):
    """Load checkpoint.

    Note:
        This function add some useless ops to the graph. It is better
        to use tf.train.init_from_checkpoint(...).
    """
    print(' [*] Loading checkpoint...')
    if os.path.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    restorer = tf.train.Saver(var_list)
    restorer.restore(session, ckpt_dir_or_file)
    print(' [*] Loading succeeds! Copy variables from % s' % ckpt_dir_or_file)


def squared_dist(A):
    expanded_a = tf.expand_dims(A, 2)
    expanded_b = tf.expand_dims(A, 1)
    l1 = tf.reduce_sum(expanded_a - expanded_b, 3)
    l2 = tf.sqrt(tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 3))
    return l1, l2

