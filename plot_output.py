

import h5py
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import paths

CROP_BORDER_DIM = 15


class Plotter():

    def __init__(self):
        self.figure = plt.figure(0)
        self.subplots = []

    def add_subplot(self, title, img):
        self.subplots.append((title, img))

    def show(self):
        num_subplots = len(self.subplots)
        y_dim = 3.0
        x_dim = math.ceil(num_subplots/y_dim)

        for i in range(len(self.subplots)):
            title, img = self.subplots[i]

            print "plotting: " + str(title)
            print img.shape

            plt.subplot(x_dim, y_dim, i + 1)
            plt.title(title)
            plt.imshow(img)

        plt.show()


def main():

    dataset_file = paths.choose_from(paths.HEATMAPS_DATASET_DIR)
    dataset = h5py.File(paths.HEATMAPS_DATASET_DIR + dataset_file)

    for i in range(dataset['rgbd_data'].shape[0]):
        rgbd_img = dataset['rgbd_data'][i]
        heatmaps = dataset['heatmaps'][i]
        indepent_grasp_points = dataset["independent_grasp_points"][i]
        convolved_heatmaps = dataset['convolved_heatmaps'][i]
        dependent_grasp_points = dataset["dependent_grasp_points"][i]

        plotter = Plotter()

        plotter.add_subplot('rgb', rgbd_img[:, :, 0:3])
        plotter.add_subplot('d', rgbd_img[:, :, 3])
        plotter.add_subplot('d', rgbd_img[:, :, 3])

        plotter.add_subplot('l_obs', heatmaps[:, :, 0])
        plotter.add_subplot('p_obs', heatmaps[:, :, 1])
        plotter.add_subplot('r_obs', heatmaps[:, :, 2])

        plotter.add_subplot('l_independent', indepent_grasp_points[0, :, :, :])
        plotter.add_subplot('p_independent', indepent_grasp_points[1, :, :, :])
        plotter.add_subplot('r_independent', indepent_grasp_points[2, :, :, :])

        plotter.add_subplot('l_convolved', convolved_heatmaps[:, :, 0])
        plotter.add_subplot('p_convolved', convolved_heatmaps[:, :, 1])
        plotter.add_subplot('r_convolved', convolved_heatmaps[:, :, 2])

        plotter.add_subplot('l_dependent', dependent_grasp_points[0, :, :, :])
        plotter.add_subplot('p_dependent', dependent_grasp_points[1, :, :, :])
        plotter.add_subplot('r_dependent', dependent_grasp_points[2, :, :, :])

        plotter.show()

        # fig = plt.figure(2)
        # ax = fig.gca(projection='3d')
        # X = []
        # Y = []
        # Z = []
        #
        # for x in range(heatmaps.shape[0]):
        #     for y in range(heatmaps.shape[1]):
        #         X.append(x)
        #         Y.append(y)
        #         Z.append(heatmaps[x, y, 0])
        #
        # #X, Y = np.meshgrid(X, Y)
        # X
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        #         linewidth=0, antialiased=False)
        # ax.set_zlim(-1.01, 1.01)
        #
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        #
        # plt.show()

if __name__ == '__main__':
    main()





