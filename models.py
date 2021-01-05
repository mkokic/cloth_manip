import tensorflow as tf
from tf_utils import counter
from networks import *

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = 200000.
BN_DECAY_CLIP = 0.99

BASE_LEARNING_RATE = 0.001
MOMENTUM = 0.9
DECAY_STEP = 200000.
DECAY_RATE = 0.7


def get_bn_decay(opt, batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * opt.bs,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def get_learning_rate(opt, batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * opt.bs,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


class Model:
    def __init__(self, opt):

        self.plh = {
            # 'pInit': tf.placeholder(tf.float32, shape=(None, opt.num * 3)),
            'pInit': tf.placeholder(tf.float32, shape=(None, 32, 32, 32)),
            'gIndex': tf.placeholder(tf.float32, shape=(None, 6)),
            'pFinal': tf.placeholder(tf.float32, shape=(None, opt.num * 3)),
            'pError': tf.placeholder(tf.float32, shape=(None, 1)),
            'eval': tf.placeholder(tf.bool, shape=())
        }

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        self.sess = tf.Session(config=config)
        self.cnt = counter()

        with tf.device('/gpu:0'):
            # Get pose and shape
            if opt.problem == 'posh':
                # PointNet
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(opt, batch)

                # self.pred_grasp, self.pred_pts, self.predictions = cnn_posh(opt, self.plh['pFinal'])
                self.pred_grasp, self.pred_pts, self.predictions = cnn_posh_pointnet(opt, self.plh['pFinal'],
                                                                                     self.plh['eval'], bn_decay)
                # self.loss_pts = tf.losses.mean_squared_error(self.plh['pInit'], self.pred_pts)
                self.loss_pts = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(self.plh['pInit'], self.pred_pts, 20))
                self.loss_grasp = tf.losses.mean_squared_error(self.plh['gIndex'], self.pred_grasp)

                learning_rate = get_learning_rate(opt, batch)
                # self.optim = tf.train.AdamOptimizer(opt.lr).minimize(self.loss_pts)
                self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_pts, global_step=batch)
                self.optim_grasp = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_grasp)

                tf.summary.scalar('loss', self.loss_pts)
                self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Get heatmap
            if opt.problem == 'heatmap':
                self.pred_score = cnn_heatmap(opt, self.plh['pInit'], self.plh['gIndex'], self.plh['eval'])
                self.loss_heatmap = tf.losses.mean_squared_error(self.plh['pError'], self.pred_score)
                self.optim = tf.train.AdamOptimizer(opt.lr).minimize(self.loss_heatmap)

                tf.summary.scalar('loss', self.loss_heatmap)
                self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
