import tensorflow as tf
import time
from sklearn.metrics import mean_squared_error as mse
import numpy as np
from IPython import embed
from tf_utils import counter, load_checkpoint
from vis_utils import *
from data_utils import *
from train_options import TrainOptions
from models import Model
import os


class ClothModel(object):
    def __init__(self):
        opt = TrainOptions().parse()
        os.system('rm -rf ./logs_' + opt.problem)
        dataset = load_data_cloth(opt.num)
        print('Dataset loaded!')
        model = Model(opt)

        if opt.mode == 'train':
            os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
            self.train(opt, dataset, model)
            self.test(opt, dataset, model)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            self.test(opt, dataset, model)

    @staticmethod
    def train(opt, data, model):
        sess = model.sess
        cnt_var = model.cnt
        plh = [model.plh['pInit'],
               model.plh['gIndex'],
               model.plh['pFinal'],
               model.plh['pError'],
               model.plh['eval']]

        merged = tf.summary.merge_all()
        if opt.problem == 'posh':
            log_path = 'logs_posh'
        else:
            log_path = 'logs_heatmap'
        dtr = [data[0]['pInit'],
               data[0]['gIndex'],
               data[0]['pFinal'],
               data[0]['pError']]
        dte = [data[1]['pInit'],
               data[1]['gIndex'],
               data[1]['pFinal'],
               data[1]['pError']]

        train_writer = tf.summary.FileWriter(log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_path + '/test')
        tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        if opt.cont: load_checkpoint('models_' + str(opt.problem), sess)

        epoch_final = sess.run(cnt_var[0]) + opt.epochs
        for epoch in range(epoch_final - opt.epochs, epoch_final):
            sess.run(cnt_var[1])

            losses_tr = []
            epoch_start_time = time.time()

            for iter in range(dtr[0].shape[0] // opt.bs):
                train_batch = next(get_batch(dtr, opt.bs))
                # aug_data = jitter_point_cloud(rotate_point_cloud(np.array(train_batch[2]).reshape((-1, opt.num, 3))))
                feed_dict = dict()
                feed_dict[plh[0]] = np.array(train_batch[0]).reshape((-1, opt.num * 3))
                feed_dict[plh[1]] = np.array(train_batch[1]).reshape((-1, 3))
                # feed_dict[plh[2]] = aug_data.reshape((-1, opt.num * 3))
                feed_dict[plh[2]] = np.array(train_batch[2]).reshape((-1, opt.num * 3))
                feed_dict[plh[3]] = np.array(train_batch[3]).reshape((-1, 1))
                feed_dict[plh[-1]] = True

                loss_tr_, _ = sess.run([model.loss, model.optim], feed_dict)
                losses_tr.append(loss_tr_)
                summary = sess.run(merged, feed_dict)
                train_writer.add_summary(summary, epoch)

            feed_dict = {plh[i]: dte[i] for i in range(4)}
            feed_dict[plh[-1]] = False
            losses_te = sess.run(model.loss, feed_dict)
            summary = sess.run(merged, feed_dict)
            test_writer.add_summary(summary, epoch)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print(
                '[%d/%d] - ptime: %.2f, loss_train: %.6f, loss_test: %.6f'
                % ((epoch + 1), epoch_final, per_epoch_ptime, np.mean(losses_tr), losses_te))
            if (epoch + 1) % 100 == 0:
                saver.save(sess, '%s/Epoch_(%d).ckpt' % ('models_' + str(opt.problem), epoch))

        save_path = saver.save(sess, '%s/Epoch_(%d).ckpt' % ('models_' + str(opt.problem), epoch))
        print("Model saved in file: %s" % save_path)
        sess.close()

    @staticmethod
    def test(opt, data, model):
        sess = tf.InteractiveSession()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        load_checkpoint('models_' + str(opt.problem), sess)
        plh = [model.plh['pInit'],
               model.plh['gIndex'],
               model.plh['pFinal'],
               model.plh['pError'],
               model.plh['eval']]

        tid = 1

        if opt.load_raw:
            f_raw = h5py.File('cloth_predicted.hdf5', 'r')
            p_init = np.array(f_raw.get('pInit')).reshape((-1, 3))
            g_init = np.array(f_raw.get('gIndex')).reshape((3,))
            f_init = np.array(f_raw.get('pFinal')).reshape((-1, 3))
            dte = [p_init.reshape(-1, opt.num * 3),
                   g_init.reshape(-1, 3),
                   f_init.reshape(-1, opt.num * 3),
                   np.zeros((len(p_init), 1))]
        else:
            dte = [data[tid]['pInit'],
                   data[tid]['gIndex'],
                   data[tid]['pFinal'],
                   data[tid]['pError']]

        feed_dict = {plh[i]: dte[i] for i in range(4)}
        feed_dict[plh[-1]] = False
        if opt.problem == 'posh':
            grasp, pts = sess.run([model.pred_grasp, model.pred_pts], feed_dict)
            pts = pts.reshape((len(pts), -1))
        else:
            score = sess.run(model.pred_score, feed_dict).reshape(-1, 1)

        for k in range(dte[0].shape[0]):
            if opt.problem == 'posh':
                # Display error
                print((pts[k].reshape(-1, 3) - dte[0][k].reshape(-1, 3)) ** 2).mean()
                # Plot predicted shape
                plot(dte[0][k].reshape(-1, 3),
                     dte[1][k].reshape(-1, 3),
                     dte[2][k].reshape(-1, 3),
                     pts[k].reshape(-1, 3),
                     grasp[k].reshape(-1, 3))

                raw = raw_input('Save as hdf5 [y/n]? ')
                if raw == 'y':
                    f_pred = h5py.File(
                        '/mnt/md0/mkokic/Github_Mia/cloth_manip/cloth_predicted.hdf5', 'w')
                    pInit_ = f_pred.create_dataset("pInit",
                                                   (pts[k].reshape(-1, 3).shape[0], pts[k].reshape(-1, 3).shape[1]),
                                                   dtype='f8')
                    pInit_[...] = pts[k].reshape(-1, 3)
                    gIndex_ = f_pred.create_dataset("gIndex", (
                        dte[1][k].reshape(-1, 3).shape[0], dte[1][k].reshape(-1, 3).shape[1]),
                                                    dtype='f8')
                    gIndex_[...] = dte[1][k].reshape(-1, 3)
                    pFinal_ = f_pred.create_dataset("pFinal", (
                        dte[2][k].reshape(-1, 3).shape[0], dte[2][k].reshape(-1, 3).shape[1]),
                                                    dtype='f8')
                    pFinal_[...] = dte[2][k].reshape(-1, 3)

            if opt.problem == 'heatmap':
                embed()


if __name__ == '__main__':
    ClothModel()
