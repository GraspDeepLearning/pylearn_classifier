

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
        heatmaps = dataset['heatmaps'][i]
        convolved_heatmaps = dataset['convolved_heatmaps'][i]
        palm_conv = dataset["p_convolved_heatmaps"][i]
        l_conv = dataset["l_convolved_heatmaps"][i]
        r_conv = dataset["r_convolved_heatmaps"][i]

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

        palm_out = convolved_heatmaps[:, :, 1]
        palm_min = np.argmin(palm_out)
        palm_out_x, palm_out_y = (palm_min / palm_out.shape[1], palm_min % palm_out.shape[1])

        l_out = convolved_heatmaps[:, :, 0]
        l_min = np.argmin(l_out)
        l_out_x, l_out_y = (l_min / l_out.shape[1], l_min % l_out.shape[1])

        r_out = convolved_heatmaps[:, :, 2]
        r_min = np.argmin(r_out)
        r_out_x, r_out_y = (r_min / r_out.shape[1], r_min % r_out.shape[1])

        rgbd_overlay = np.copy(rgbd_img)
        x_offset = (rgbd_overlay.shape[0]-palm_out.shape[0])/2
        y_offset = (rgbd_overlay.shape[1]-palm_out.shape[1])/2

        rgbd_overlay[x_offset+ l_out_x-10:x_offset+l_out_x+10,y_offset+ l_out_y-10:y_offset+l_out_y+10, :] = 0
        rgbd_overlay[x_offset+ palm_out_x-10:x_offset+palm_out_x+10,y_offset+ palm_out_y-10:y_offset+palm_out_y+10, :] = 0
        rgbd_overlay[x_offset+ r_out_x-10:x_offset+r_out_x+10,y_offset+ r_out_y-10:y_offset+r_out_y+10, :] = 0


        plotter1 = Plotter(1)

        if rgbd_img.shape[-1] == 4:
            plotter2.add_subplot('rgb', rgbd_img[:, :, 0:3])
            plotter2.add_subplot('d', rgbd_img[:, :, 3])
        else:
            plotter2.add_subplot('d', rgbd_img[:,:,0])
        if rgbd_img.shape[-1] == 4:
            plotter2.add_subplot("convolved_overlay", rgbd_overlay[:, :, :-1])
        else:
            plotter2.add_subplot("convolved_overlay", rgbd_overlay[:, :, 0])

        #plotter1.show()
        plotter2.show()
        plt.show()



if __name__ == '__main__':
    main()





