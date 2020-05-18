import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib


def plot_single(input):
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    # ax1.set_xlim([0, 50])
    # ax1.set_ylim([0, 50])
    # ax1.set_zlim([0, 50])
    ax1.scatter(input[:, 0], input[:, 1], input[:, 2], c='gray', marker='.', alpha=1.0)
    plt.show()


def plot(inputs, grasp, output, output_pred, grasp_pred):
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    # ax1.set_xlim([inputs[:, 0].min(), inputs[:, 0].max()])
    # ax1.set_ylim([inputs[:, 1].min(), inputs[:, 1].max()])
    # ax1.set_zlim([inputs[:, 2].min(), inputs[:, 2].max()])
    # ax2.set_xlim([output[:, 0].min(), output[:, 0].max()])
    # ax2.set_ylim([output[:, 1].min(), output[:, 1].max()])
    # ax2.set_zlim([output[:, 2].min(), output[:, 2].max()])
    # ax3.set_xlim([output[:, 0].min(), output[:, 0].max()])
    # ax3.set_ylim([output[:, 1].min(), output[:, 1].max()])
    # ax3.set_zlim([output[:, 2].min(), output[:, 2].max()])

    # ax1.set_zlim([20, 30])
    # ax2.set_zlim([20, 30])
    # ax3.set_zlim([20, 30])

    ax1.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c='gray', marker='.', alpha=0.7)
    ax1.scatter(grasp[:, 0], grasp[:, 1], grasp[:, 2], c='black', marker='x', alpha=1.0, s=100)

    ax2.scatter(output[:, 0], output[:, 1], output[:, 2], c='red', marker='.', alpha=0.7)
    # ax2.scatter(grasp[:, 0], grasp[:, 1], grasp[:, 2], c='black', marker='x', alpha=1.0, s=100)

    ax3.scatter(output_pred[:, 0], output_pred[:, 1], output_pred[:, 2], c='blue', marker='.', alpha=0.7)
    # ax3.scatter(grasp_pred[:, 0], grasp_pred[:, 1], grasp_pred[:, 2], c='black', marker='x', alpha=1.0, s=100)

    plt.show()
