import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib


def plot_single(input, limit=False):
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    if limit:
        ax1.set_xlim([0, 32])
        ax1.set_ylim([0, 32])
        ax1.set_zlim([0, 32])
    ax1.scatter(input[:, 0], input[:, 1], input[:, 2], c='gray', marker='.', alpha=1.0)
    plt.show()


def plot(inputs, grasp, output, output_pred, grasp_pred):
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    ax2.set_xlim([0, 32])
    ax2.set_ylim([0, 32])
    ax2.set_zlim([0, 32])
    ax3.set_xlim([0, 32])
    ax3.set_ylim([0, 32])
    ax3.set_zlim([0, 32])

    # ax2.set_xlim([inputs.min(0)[0], inputs.max(0)[0]])
    # ax2.set_ylim([inputs.min(0)[1], inputs.max(0)[1]])
    # ax2.set_zlim([inputs.min(0)[2], inputs.max(0)[2]])
    # ax3.set_xlim([output_pred.min(0)[0], output_pred.max(0)[0]])
    # ax3.set_ylim([output_pred.min(0)[1], output_pred.max(0)[1]])
    # ax3.set_zlim([output_pred.min(0)[2], output_pred.max(0)[2]])

    ax1.scatter(output[:, 0], output[:, 1], output[:, 2], c='red', marker='.', alpha=0.7)
    # ax1.scatter(grasp[:, 0], grasp[:, 1], grasp[:, 2], c='black', marker='x', alpha=1.0, s=100)

    ax2.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c='gray', marker='.', alpha=0.7)
    # ax2.scatter(grasp[:, 0], grasp[:, 1], grasp[:, 2], c='black', marker='x', alpha=1.0, s=100)

    ax3.scatter(output_pred[:, 0], output_pred[:, 1], output_pred[:, 2], c='blue', marker='.', alpha=0.7)
    # ax3.scatter(grasp_pred[:, 0], grasp_pred[:, 1], grasp_pred[:, 2], c='black', marker='x', alpha=1.0, s=100)

    plt.show()
