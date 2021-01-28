import tensorflow as tf
import time
import numpy as np
from tf_utils import counter, load_checkpoint
from vis_utils import *
from data_utils import *
from train_options import TrainOptions
from models import Model
import os
from IPython import embed


class ClothModel(object):
    def __init__(self):
        opt = TrainOptions().parse()
        os.system('rm -rf ./logs_' + opt.problem)
        dataset = load_data(opt.num)
        print('Dataset loaded!')
        model = Model(opt)

        if opt.mode == 'train':
            self.train(opt, dataset, model)
            self.test(opt, dataset, model)
        else:
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
        # tf.global_variables_initializer().run()

        saver = tf.train.Saver()
        if opt.cont: load_checkpoint('models_' + str(opt.problem), sess)

        epoch_final = sess.run(cnt_var[0]) + opt.epochs
        for epoch in range(epoch_final - opt.epochs, epoch_final):
            sess.run(cnt_var[1])

            losses_tr_pts, losses_tr_grasp = [], []
            epoch_start_time = time.time()

            for iter in range(dtr[0].shape[0] // opt.bs):
                train_batch = next(get_batch(dtr, opt.bs))
                aug_data = rotate_point_cloud(jitter_point_cloud(np.array(train_batch[2]).reshape((-1, opt.num, 3))))
                feed_dict = dict()
                # if opt.problem == 'heatmap':
                #     feed_dict[plh[0]] = aug_data.reshape((-1, opt.num * 3))
                # else:
                feed_dict[plh[0]] = np.array(train_batch[0]).reshape((-1, 32, 32, 32))
                feed_dict[plh[1]] = np.array(train_batch[1]).reshape((-1, 6))
                feed_dict[plh[2]] = np.array(aug_data).reshape((-1, opt.num * 3))
                feed_dict[plh[3]] = np.array(train_batch[3]).reshape((-1, 1))
                feed_dict[plh[-1]] = True

                if opt.problem == 'posh':
                    loss_tr_pts_, loss_tr_grasp_, _, __ = sess.run(
                        [model.loss_pts, model.loss_grasp, model.optim, model.optim_grasp], feed_dict)
                    losses_tr_pts.append(loss_tr_pts_)
                    losses_tr_grasp.append(loss_tr_grasp_)
                else:
                    loss_tr_pts_, _ = sess.run([model.loss_heatmap, model.optim], feed_dict)
                    losses_tr_pts.append(loss_tr_pts_)
                summary = sess.run(merged, feed_dict)
                train_writer.add_summary(summary, epoch)

            feed_dict = {plh[i]: dte[i] for i in range(4)}
            feed_dict[plh[-1]] = False

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            if opt.problem == 'posh':
                losses_te_pts, losses_te_grasp = sess.run([model.loss_pts, model.loss_grasp], feed_dict)
                print(
                    '[%d/%d] - ptime: %.2f, loss_tr_pts: %.6f, loss_tr_grasp: %.6f, '
                    'loss_te_pts: %.6f, loss_te_grasp: %.6f'
                    % ((epoch + 1), epoch_final, per_epoch_ptime, np.mean(losses_tr_pts), np.mean(losses_tr_grasp),
                       losses_te_pts, losses_te_grasp))
            else:
                losses_te_pts = sess.run(model.loss_heatmap, feed_dict)
                print(
                    '[%d/%d] - ptime: %.2f, loss_train: %.6f, loss_test: %.6f'
                    % ((epoch + 1), epoch_final, per_epoch_ptime, np.mean(losses_tr_pts), losses_te_pts))
            summary = sess.run(merged, feed_dict)
            test_writer.add_summary(summary, epoch)

            if (epoch + 1) % 50 == 0:
                saver.save(sess, '%s/Epoch_(%d).ckpt' % ('models_' + str(opt.problem), epoch))

        save_path = saver.save(sess, '%s/Epoch_(%d).ckpt' % ('models_' + str(opt.problem), epoch))
        print("Model saved in file: %s" % save_path)
        sess.close()

    @staticmethod
    def test(opt, data, model):
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # sess = tf.InteractiveSession()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        load_checkpoint('models_' + str(opt.problem), sess)
        plh = [model.plh['pInit'],
               model.plh['gIndex'],
               model.plh['pFinal'],
               model.plh['pError'],
               model.plh['eval']]

        tid = 1
        dte = [data[tid]['pInit'],
               data[tid]['gIndex'],
               data[tid]['pFinal'],
               data[tid]['pError']]

        feed_dict = {plh[i]: dte[i] for i in range(4)}
        feed_dict[plh[-1]] = False
        if opt.problem == 'posh':
            grasp, pts = sess.run([model.pred_grasp, model.predictions], feed_dict)
            # pts = pts.reshape((len(pts), -1))
        else:
            scores = sess.run(model.pred_score, feed_dict).reshape(-1, 1)

        pts_err_all = []
        for k in range(dte[0].shape[0]):
            if opt.problem == 'posh':
                # Display points and grasp error
                # pts_err = ((pts[k].reshape(-1, 3)[:, [0, 2]] - dte[0][k].reshape(-1, 3)[:, [0, 2]]) ** 2).mean()
                # pts_err_all.append(pts_err)
                # grasp_err = ((grasp[k].reshape(-1, 3) - dte[1][k].reshape(-1, 3)) ** 2).mean()
                # print('Points error: %.5f, Grasp error: %.5f' % (pts_err, grasp_err))
                # Plot predicted shape
                # grasp[:, [1, 4]] = 0
                pts_gt = np.array(dte[0][k].nonzero()).transpose()
                tmp = pts[k]
                tmp[tmp < 0.5] = 0.0
                tmp[tmp >= 0.5] = 1.0
                pts_pred = np.array(tmp.nonzero()).transpose()

                plot(pts_gt,
                     dte[1][k].reshape(-1, 3),
                     dte[2][k].reshape(-1, 3),
                     pts_pred,
                     grasp[k].reshape(-1, 3))
                embed()

                # # Reset graph and het heatmp
                # tf.reset_default_graph()
                # opt.problem = 'heatmap'
                # model = Model(opt)
                # plh = [model.plh['pInit'],
                #        model.plh['gIndex'],
                #        model.plh['pFinal'],
                #        model.plh['pError'],
                #        model.plh['eval']]
                # model.sess = tf.InteractiveSession()
                # model.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                # load_checkpoint('models_' + str(opt.problem), model.sess)
                # # Create grasps
                # pred_cloth = np.array(pts[k]).reshape((opt.num, 3))
                # # pred_cloth_grasps = list()
                # # for p1 in range(0, opt.num, 200):
                # #     for p2 in range(0, opt.num, 200):
                # #         pred_cloth_grasps.append(list(pred_cloth[p1]) + list(pred_cloth[p2]))
                #
                # # Alternatively run ground truth grasps to check accuracy
                # pred_cloth_grasps = dte[1][k]
                #
                # # Run predicted cloth and grasps through the heatmap network
                # feed_dict = dict()
                # # feed_dict[plh[0]] = np.repeat(pred_cloth, len(pred_cloth_grasps)).reshape((-1, opt.num * 3))
                # feed_dict[plh[0]] = pred_cloth.reshape((-1, opt.num * 3))
                # feed_dict[plh[1]] = np.array(pred_cloth_grasps).reshape((-1, 6))
                # feed_dict[plh[-1]] = False
                # # Get predicted grasp scores
                # scores = model.sess.run(model.pred_score, feed_dict).reshape(-1, )
                # print('Predicted grasp score: %.5f, Real grasp score: %.5f' % (scores[0], dte[-1][k]))
                # # print('Predicted grasp score: %.5f' % scores)
                # opt.problem = 'posh'
                # embed()

            if opt.problem == 'heatmap':
                print('Predicted grasp score: %.5f, Real grasp score: %.5f' % (scores[k], dte[-1][k]))

        print(np.array(pts_err_all).mean())


if __name__ == '__main__':
    ClothModel()
