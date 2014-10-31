

import h5py
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import choose
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
        y_dim = 2.0
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

    dataset_file = choose.choose_from(paths.HEATMAPS_DATASET_DIR)
    dataset = h5py.File(paths.HEATMAPS_DATASET_DIR + dataset_file)

    for i in range(dataset['rgbd_data'].shape[0]):
        rgbd_img = dataset['rgbd_data'][i]
        heatmaps = dataset['normalized_heatmaps'][i]
        #indepent_grasp_points = dataset["independent_grasp_points"][i]
        #convolved_heatmaps = dataset['convolved_heatmaps'][i]
        #dependent_grasp_points = dataset["dependent_grasp_points"][i]
        #palm_conv = dataset["p_convolved_heatmaps"][i]
        #l_conv = dataset["l_convolved_heatmaps"][i]
        #r_conv = dataset["r_convolved_heatmaps"][i]
        #best_grasp = dataset["best_grasp"][i]

        plotter1 = Plotter(1)

        plotter1.add_subplot('grey_scale', rgbd_img[:, :, 0])
        #plotter1.add_subplot('output', heatmaps[:,:,0])
        #plotter1.add_subplot('best_grasp', best_grasp)


        # p_to_f_dist = 10
        # l = heatmaps[:, p_to_f_dist*2:, 0]
        # p = heatmaps[:, p_to_f_dist:-p_to_f_dist, 1]
        # r = heatmaps[:, :-p_to_f_dist*2, 2]
        #
        # out = l * p * r
        #
        # plotter1.add_subplot("out", out)
        #
        out = heatmaps[:, :, 0]
        out_min = np.argmin(out)
        out_x, out_y = (out_min / out.shape[1], out_min % out.shape[1])
        out_min_plot = np.copy(out)
        marker_size=1
        out_min_plot[out_x-marker_size:out_x+marker_size, out_y-marker_size:out_y+marker_size] = 0
        plotter1.add_subplot("out_min", out_min_plot)

        out = heatmaps[:, :, 0]
        out_max = np.argmax(out)
        out_x, out_y = (out_max / out.shape[1], out_max % out.shape[1])
        out_max_plot = np.copy(out)
        marker_size=1
        out_max_plot[out_x-marker_size:out_x+marker_size, out_y-marker_size:out_y+marker_size] = 255
        plotter1.add_subplot("out_max", out_max_plot)

        plotter1.show()

        plt.show()



if __name__ == '__main__':
    main()





