import numpy as np
from IPython import embed
import random
from cloth import Cloth
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib


def plot_single(input):
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_xlim([0, 500])
    ax1.set_ylim([-500, 0])
    ax1.set_zlim([0, 500])
    ax1.scatter(input[:, 0], input[:, 1], input[:, 2], c='gray', marker='.', alpha=1.0)
    plt.show()


def main():
    timeStep = int(1000 / 60.)

    num_list = np.arange(1000)

    dims = [(i, j) for i in num_list for j in num_list if i * j == 1000]

    for xdim, ydim in dims:
        mass = int(5000 / (xdim * ydim))
        steps = 100
        cloth = Cloth(xdim, ydim, xdim, ydim, mass, 0.01)

        print('Cloth created!')

        pInit = np.array([cloth.particles[k].pos for k in range(cloth.pN)]).reshape(1, cloth.pN * 3)
        # plot_single(pInit[0].reshape(-1, 3))
        # embed()
        pFinal = np.zeros((1, cloth.pN * 3))

        for grasp_id in range(1, 1000, 100):
            cloth.particles[grasp_id].isStatic = True
            for ts in range(steps):
                finalPos = cloth._step(timeStep)
                if ts == steps - 1:
                    pFinal = finalPos.reshape(1, cloth.pN * 3)
            print('Timestep %d/%d' % (ts, steps))

            gIndex = pInit.reshape(cloth.pN, 3)[grasp_id].reshape(1, 3)

            f_train = h5py.File(
                '/mnt/md0/mkokic/Github_Mia/cloth_manip/create_cloth/hdf5/cloth_' + str(xdim) + 'x' + str(
                    ydim) + '_' + str(
                    grasp_id) + '.hdf5', 'w')
            pInit_ = f_train.create_dataset("pInit", (pInit.shape[0], pInit.shape[1]),
                                            dtype='f8')
            pInit_[...] = pInit
            gIndex_ = f_train.create_dataset("gIndex", (gIndex.shape[0], gIndex.shape[1]),
                                             dtype='f8')
            gIndex_[...] = gIndex
            pFinal_ = f_train.create_dataset("pFinal", (pFinal.shape[0], pFinal.shape[1]),
                                             dtype='f8')
            pFinal_[...] = pFinal


if __name__ == '__main__':
    main()
