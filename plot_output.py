

import h5py
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


import paths

CROP_BORDER_DIM = 15


class format_subplot():

    def __init__(self, ax, img):
        self.ax = ax
        self.img = img
        self.ax.format_coord = self.format_coord

    def format_coord(self, x, y):

        return "x=" + str(x) + "  y=" + str(y) + "  z=" + str(self.img[int(y),int(x)])


class Plotter():

    def __init__(self, figure_num=0):
        self.figure_num = figure_num
        self.subplots = []
        self.histograms = []

    def add_subplot(self, title, img):
        self.subplots.append((title, img))

    def add_histogram(self, title, img):
        self.histograms.append((title, img))

    def show(self):
        figure = plt.figure(self.figure_num)
        num_histograms = len(self.histograms)
        num_subplots = len(self.subplots)
        y_dim = 4.0
        x_dim = math.ceil((num_subplots + num_histograms)/y_dim)

        for i in range(len(self.subplots)):
            title, img = self.subplots[i]

            print "plotting: " + str(title)
            print img.shape

            ax = plt.subplot(x_dim, y_dim, i + 1)
            format_subplot(ax, img)
            plt.title(title)
            plt.imshow(img)

        for i in range(len(self.histograms)):
            title, img = self.histograms[i]

            print "plotting: " + str(title)
            print img.shape

            plt.subplot(x_dim,y_dim, num_subplots + i + 1)
            plt.title(title)
            plt.hist(img, bins=10, alpha=0.5)

        #plt.show()


def main():

    dataset_file = paths.choose_from(paths.HEATMAPS_DATASET_DIR)
    dataset = h5py.File(paths.HEATMAPS_DATASET_DIR + dataset_file)

    for i in range(dataset['rgbd_data'].shape[0]):
        rgbd_img = dataset['rgbd_data'][i]
        heatmaps = dataset['normalized_heatmaps'][i]
        indepent_grasp_points = dataset["independent_grasp_points"][i]
        convolved_heatmaps = dataset['convolved_heatmaps'][i]
        dependent_grasp_points = dataset["dependent_grasp_points"][i]
        palm_conv = dataset["p_convolved_heatmaps"][i]
        l_conv = dataset["l_convolved_heatmaps"][i]
        r_conv = dataset["r_convolved_heatmaps"][i]

        plotter1 = Plotter(1)

        plotter1.add_subplot('rgb', rgbd_img[:, :, 0:3])
        plotter1.add_subplot('d', rgbd_img[:, :, 3])

        plotter2 = Plotter(2)

        plotter2.add_subplot('l_obs', heatmaps[:, :, 0]/heatmaps[:,:,0].max())
        plotter2.add_subplot('l_g_p * palm', palm_conv[0, :, :])
        plotter2.add_subplot('l_g_r * r_grip', r_conv[1, :, :])
        plotter2.add_subplot('l_convolved', convolved_heatmaps[:, :, 0])

        plotter2.add_subplot('p_obs', heatmaps[:, :, 1]/heatmaps[:,:,1].max())
        plotter2.add_subplot('p_g_l * l_grip', l_conv[2, :, :])
        plotter2.add_subplot('p_g_r * r_grip', r_conv[3, :, :])
        plotter2.add_subplot('p_convolved', convolved_heatmaps[:, :, 1])

        plotter2.add_subplot('r_obs', heatmaps[:, :, 2]/heatmaps[:,:,2].max())
        plotter2.add_subplot('r_g_p * palm', palm_conv[5, :, :])
        plotter2.add_subplot('r_g_l * l_grip', l_conv[4, :, :])
        plotter2.add_subplot('r_convolved', convolved_heatmaps[:, :, 2])

        #plotter.add_histogram('l_obs', heatmaps[:, :, 0]/heatmaps[:,:,0].max())
        #plotter.add_histogram('p_obs', heatmaps[:, :, 1]/heatmaps[:,:,1].max())
        #plotter.add_histogram('r_obs', heatmaps[:, :, 2]/heatmaps[:,:,2].max())

        #plotter.add_subplot('l_g_p', palm_conv[0, :, :])
        #plotter.add_subplot('l_g_r', palm_conv[1, :, :])
        #plotter.add_subplot('p_g_l', palm_conv[2, :, :])
        #plotter.add_subplot('p_g_r', palm_conv[3, :, :])
        #plotter.add_subplot('r_g_l', palm_conv[4, :, :])
        #plotter.add_subplot('r_g_p', palm_conv[5, :, :])

        #plotter.add_subplot('l_independent', indepent_grasp_points[0, :, :, :])
        #plotter.add_subplot('p_independent', indepent_grasp_points[1, :, :, :])
        #plotter.add_subplot('r_independent', indepent_grasp_points[2, :, :, :])

        #plotter.add_subplot('l_dependent', dependent_grasp_points[0, :, :, :])
        #plotter.add_subplot('p_dependent', dependent_grasp_points[1, :, :, :])
        #plotter.add_subplot('r_dependent', dependent_grasp_points[2, :, :, :])

        plotter1.show()
        plotter2.show()
        plt.show()



if __name__ == '__main__':
    main()





