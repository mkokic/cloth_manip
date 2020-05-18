import tensorflow as tf
from tf_utils import counter
from networks import *

# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = 200000.
# BN_DECAY_CLIP = 0.99
#
# BASE_LEARNING_RATE = 0.001
# MOMENTUM = 0.9
# DECAY_STEP = 200000.
# DECAY_RATE = 0.7
# def get_bn_decay(opt, batch):
#     bn_momentum = tf.train.exponential_decay(
#                       BN_INIT_DECAY,
#                       batch*opt.bs,
#                       BN_DECAY_DECAY_STEP,
#                       BN_DECAY_DECAY_RATE,
#                       staircase=True)
#     bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
#     return bn_decay
# def get_learning_rate(opt, batch):
#     learning_rate = tf.train.exponential_decay(
#                         BASE_LEARNING_RATE,  # Base learning rate.
#                         batch * opt.bs,  # Current index into the dataset.
#                         DECAY_STEP,          # Decay step.
#                         DECAY_RATE,          # Decay rate.
#                         staircase=True)
#     learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
#     return learning_rate


class Model:
    def __init__(self, opt):

        self.plh = {
            'pInit': tf.placeholder(tf.float32, shape=(None, opt.num * 3)),
            'gIndex': tf.placeholder(tf.float32, shape=(None, 3)),
            'pFinal': tf.placeholder(tf.float32, shape=(None, opt.num * 3)),
            'pError': tf.placeholder(tf.float32, shape=(None, 1)),
            'eval': tf.placeholder(tf.bool, shape=())
        }

        self.sess = tf.InteractiveSession()
        self.cnt = counter()

        # Get pose and shape
        if opt.problem == 'posh':
            self.pred_grasp, self.pred_pts = cloth_cnn_posh(opt, self.plh['pFinal'])
            self.loss = tf.losses.mean_squared_error(
                self.plh['pInit'], self.pred_pts)

        # Get heatmap
        if opt.problem == 'heatmap':
            self.pred_score = cloth_cnn_heatmap(opt, self.plh['pInit'], self.plh['gIndex'])
            self.loss = tf.losses.mean_squared_error(self.plh['pError'], self.pred_score)

        tf.summary.scalar('loss', self.loss)

        self.optim = tf.train.RMSPropOptimizer(opt.lr).minimize(self.loss)
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
