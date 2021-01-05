import h5py
import numpy as np
import glob
import random
import scipy
from sklearn.preprocessing import StandardScaler
from IPython import embed
from vis_utils import *


def scale_point_cloud(batch_data):
    """ Randomly scale the point clouds to augument the dataset
        scaling is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    scaled_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        scale_factor = random.randint(0, 9) / 10.
        scale_matrix = np.eye(3)
        id = random.randint(0, 2)
        scale_matrix[0, 0] = 0.05
        shape_pc = batch_data[k, ...]
        scaled_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), scale_matrix)
    return scaled_data


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        # Mia
        from scipy.stats import special_ortho_group
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), special_ortho_group.rvs(3))
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.1, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def normalize_data(p, data):
    return ((p - data.min(0)) / (data.max(0) - data.min(0)) * 1.).astype(np.uint8)


def scale_data(s):
    if not (hasattr(s, '__len__') and len(s) >= 2):
        raise ValueError("'s' needs have a length greater one")
    dim = len(s)
    S_m = np.identity(dim + 1)
    diag = np.append(s, 1)
    np.fill_diagonal(S_m, diag)
    return S_m


def shuffle_data(data):
    perm = np.random.permutation(len(data[0]))
    data_shuffled = [x[perm] for x in data]
    return data_shuffled


def closest_unitary(A):
    V, __, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    # B = np.dot(A, np.linalg.pinv(scipy.linalg.sqrtm(np.dot(A, A.T))))
    return U


def get_batch(data, num):
    num_el = data[0].shape[0]
    while True:
        idx = np.arange(0, num_el)
        # if shuffle:
        np.random.shuffle(idx)
        current_idx = 0
        while current_idx < num_el:
            batch_idx = idx[current_idx:current_idx + num]
            current_idx += num
            data = [[x[i] for i in batch_idx] for x in data]
            yield data


def get_ordered_list(points, x):
    points = np.concatenate((points, np.arange(2000).reshape(2000, 1)), 1)
    sorted_points = sorted(points, key=lambda p: np.sqrt((p[0] - x[0]) ** 2 + (p[1] - x[1]) ** 2 + (p[2] - x[2]) ** 2))
    sp = np.array(sorted_points).reshape(2000, 4)[:, :3]
    idx = np.array(sorted_points).reshape(2000, 4)[:, -1]
    return sp, idx


def load_data_cloth(num):
    path = '/mnt/md0/mkokic/Github_Mia/cloth-bullet-extensions/bobak/hdf5/proc_data_small'
    f_all = [h5py.File(f_hdf5, 'r') for f_hdf5 in glob.glob(path + '/train/*.hdf5')]
    # f_test = [h5py.File(f_hdf5, 'r') for f_hdf5 in glob.glob(path + '/test/*.hdf5')]

    # path = '/mnt/md0/mkokic/Github_Mia/cloth-bullet-extensions/bobak/hdf5/proc_data_small/'
    # f_all = list()
    # for f_hdf5 in glob.glob(path + '/*/*.hdf5'):
    #     try:
    #         f_all.append(h5py.File(f_hdf5, 'r'))
    #     except:
    #         embed()

    f_train = f_all[:int(len(f_all) * .9)]
    # f_all = f_train + f_test

    pInit = np.zeros((len(f_all), num, 3))
    gIndex = np.zeros((len(f_all), 6))
    pFinal = np.zeros((len(f_all), num, 3))
    pError = np.zeros((len(f_all), 1))

    idx_tr = []
    idx_te = []
    for i, f in enumerate(f_all):
        p_init = np.array(f.get('pInit'))[0].reshape((-1, 3))
        f_init = np.array(f.get('pFinal'))[0].reshape((-1, 3))
        g_init = np.array(f.get('gIndex'))[0].reshape((6,))

        if p_init.shape[0] >= num:
            # ids = sorted(random.sample(range(p_init.shape[0]), num))
            # p_init = p_init[ids, :]
            # f_init = f_init[ids, :]

            pError[i] = ((p_init - f_init) ** 2).mean()
            pInit[i] = p_init
            gIndex[i] = g_init
            pFinal[i] = f_init - g_init[:3]

            if i < len(f_train):
                idx_tr.append(i)
            else:
                idx_te.append(i)

    # p_unique = npi.unique(pInit, axis=0)
    # for pu in p_unique:
    #     idx_p = [i for i, x in enumerate(pInit) if (x == pu).all()]
    #     tmp = pError[idx_p]
    #     pError[idx_p] = 1.0 - ((tmp - min(tmp)) / (max(tmp) - min(tmp)))

    data_train = {'pInit': pInit[idx_tr].reshape((len(idx_tr), num * 3)),
                  'gIndex': gIndex[idx_tr].reshape(len(idx_tr), 6),
                  'pFinal': pFinal[idx_tr].reshape((len(idx_tr), num * 3)),
                  'pError': pError[idx_tr].reshape((len(idx_tr), 1))}
    data_test = {'pInit': pInit[idx_te].reshape((len(idx_te), num * 3)),
                 'gIndex': gIndex[idx_te].reshape(len(idx_te), 6),
                 'pFinal': pFinal[idx_te].reshape((len(idx_te), num * 3)),
                 'pError': pError[idx_te].reshape((len(idx_te), 1))}

    return data_train, data_test


def load_data(num):
    # path = '/mnt/md0/mkokic/Github_Mia/cloth-bullet-extensions/bobak/hdf5/mn40/'
    # f_train = list()
    # for f_hdf5 in glob.glob(path + '200_train/*/*.hdf5'):
    #     if 'tshirt_18' not in f_hdf5:
    #         try:
    #             f_train.append(h5py.File(f_hdf5, 'r'))
    #         except:
    #             # embed()
    #             continue
    # f_test = [h5py.File(f_hdf5, 'r') for f_hdf5 in glob.glob(path + '200_train/*/*.hdf5') if 'tshirt_18' in f_hdf5]

    path = '/home/melhua/code/pointnet/data/modelnet40_ply_hdf5_2048_pts'
    # f_train = [h5py.File(f_hdf5, 'r') for f_hdf5 in glob.glob(path + '/*train*.h5')]
    # f_test = [h5py.File(f_hdf5, 'r') for f_hdf5 in glob.glob(path + '/*test*.h5')]
    f_train = [f_xyz for f_xyz in glob.glob(path + '/test/*.xyz')][:64]
    f_test = [f_xyz for f_xyz in glob.glob(path + '/test/*.xyz')][-32:]

    # random.shuffle(f_train)
    f_all = f_train + f_test

    pInit = np.zeros((1, 32, 32, 32))
    gIndex = np.zeros((1, 6))
    pFinal = np.zeros((1, num, 3))
    s = 0

    # pInit = np.zeros((len(f_all), num, 3))
    # gIndex = np.zeros((len(f_all), 6))
    # pFinal = np.zeros((len(f_all), num, 3))
    # pError = np.zeros((len(f_all), 1))
    # idx_tr = []
    # idx_te = []
    # for i, f in enumerate(f_all):
    #     try:
    #         p_init = np.array(f.get('pInit'))[0].reshape((1, -1, 3))
    #     except:
    #         continue
    #     f_init = np.array(f.get('pFinal'))[0].reshape((1, -1, 3))
    #     g_init = np.array(f.get('gIndex'))[0].reshape((1, 6))
    #
    #     try:
    #         ids = sorted(random.sample(range(p_init.shape[1]), num))
    #         p_init = p_init[:, ids, :]
    #         f_init = f_init[:, ids, :]
    #
    #         g_init = np.concatenate(((g_init[0, :3] - p_init.mean(1)) / (p_init.std(1) - 1e-8),
    #                                  (g_init[0, 3:] - p_init.mean(1)) / (p_init.std(1) - 1e-8)), 0).reshape(
    #             1, 6)
    #         p_init = (p_init - p_init.mean(1)) / (p_init.std(1) - 1e-8)
    #         f_init = (f_init - f_init.mean(1)) / (f_init.std(1) - 1e-8)
    #
    #         if f_init[0, :, :].min() > -4 and f_init[0, :, :].max() < 4:
    #             gIndex[i] = g_init[0]
    #             pInit[i] = p_init[0, :, :]
    #             pFinal[i] = f_init[0, :, :]
    #             if i < len(f_train):
    #                 idx_tr.append(i)
    #             else:
    #                 idx_te.append(i)
    #     except:
    #         continue

    for i, f in enumerate(f_all):
        # f_init = f['data'][:][:, :num, :]
        # p_init = f['data'][:][:, :num, :]
        # try:
        pts_file = np.array([pts[:-3].strip('e-').split(' ') for pts in open(f, "r").readlines()],
                            dtype=float).reshape(-1, 3)
        scaler = StandardScaler()
        pts_file = scaler.fit_transform(pts_file)
        a = np.abs(scipy.stats.zscore(pts_file))
        pts_file = pts_file[(a < 3).all(1)]
        pts_file = pts_file[:num, :]
        pts_scaled = np.array(
            (pts_file - pts_file.min(0).min()) / (pts_file.max(0).max() - pts_file.min(0).min()) * 31,
            dtype=np.uint8).reshape((-1, 3))
        p_init = np.zeros((32, 32, 32))
        p_init[pts_scaled[:, 0], pts_scaled[:, 1], pts_scaled[:, 2]] = 1
        p_init = p_init.reshape((1, 32, 32, 32))
        f_init = pts_file.reshape(1, num, 3)
        # except:
        #     continue

        g_init = np.zeros((len(f_init), 6))
        if i < len(f_train):
            s += f_init.shape[0]
        gIndex = np.vstack((gIndex, g_init))
        pInit = np.vstack((pInit, p_init))
        pFinal = np.vstack((pFinal, f_init))

    gIndex = gIndex[1:]
    pInit = pInit[1:]
    pFinal = pFinal[1:]

    idx_tr = np.arange(s)
    idx_te = np.arange(s, len(pInit))
    pError = np.zeros((len(pInit), 1))

    data_train = {'pInit': pInit[idx_tr].reshape((len(idx_tr), 32, 32, 32)),
                  'gIndex': gIndex[idx_tr].reshape(len(idx_tr), 6),
                  'pFinal': pFinal[idx_tr].reshape((len(idx_tr), num * 3)),
                  'pError': pError[idx_tr].reshape((len(idx_tr), 1))}
    data_test = {'pInit': pInit[idx_te].reshape((len(idx_te), 32, 32, 32)),
                 'gIndex': gIndex[idx_te].reshape(len(idx_te), 6),
                 'pFinal': pFinal[idx_te].reshape((len(idx_te), num * 3)),
                 'pError': pError[idx_te].reshape((len(idx_te), 1))}
    return data_train, data_test
