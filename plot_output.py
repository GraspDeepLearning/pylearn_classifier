
import os
import h5py
from scipy.signal import argrelextrema
import scipy.misc
import copy
import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import paths

CROP_BORDER_DIM = 15


class Plotter():

    def __init__(self):
        plt.figure(0)
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

# class BaseFigureGenerator():
#
#     _figure_number = 0
#
#     def __init__(self, rgbd_image, heatmaps):
#         self.rgbd_image = rgbd_image
#         self.heatmaps = heatmaps
#
#     @staticmethod
#     def get_figure_number():
#         BaseFigureGenerator._figure_number += 1
#         return BaseFigureGenerator._figure_number
#
#     def build(self):
#         pass
#
#     def plot(self):
#         pass
#
#     def run(self):
#         self.build()
#         figure = plt.figure(BaseFigureGenerator.get_figure_number())
#         self.plot(figure)
#
#
# class InputOutputFigure(BaseFigureGenerator):
#
#     def plot(self, figure):
#
#         subplot_index = 1
#
#         plt.subplot(3, 3, subplot_index)
#         subplot_index += 1
#         plt.title('rgb')
#         plt.imshow(self.rgbd_image[:, :, 0:3])
#
#         plt.subplot(3, 3, subplot_index)
#         subplot_index += 2
#         plt.title('d')
#         plt.imshow(self.rgbd_image[:, :, 3])
#
#         for i in range(self.heatmaps.shape[2]):
#             plt.subplot(3, 3, subplot_index)
#             subplot_index += 1
#             plt.title('heatmap_' + str(i))
#             plt.imshow(self.heatmaps[:, :, i])
#
#         for index in range(self.heatmaps.shape[2]):
#             heatmap = self.heatmaps[:, :, index]
#
#             #ax = plt.axes(projection='3d')
#             ax = figure.add_subplot(3, 3, subplot_index, projection='3d')
#             subplot_index += 1
#             plt.title('heatmap_' + str(index))
#
#             x = np.zeros(heatmap.shape[0]*heatmap.shape[1])
#             y = np.zeros(heatmap.shape[0]*heatmap.shape[1])
#             z = np.zeros(heatmap.shape[0]*heatmap.shape[1])
#
#             count = 0
#             for i in range(heatmap.shape[0]):
#                 for j in range(heatmap.shape[1]):
#                     x[count] = i
#                     y[count] = j
#                     z[count] = heatmap[i, j]
#                     count += 1
#
#             ax.scatter(x, y, z)
#
#
#
#
# class PriorsFigure(BaseFigureGenerator):
#     def __init__(self, rgbd_image, heatmaps, observed, priors):
#         BaseFigureGenerator.__init__(self, rgbd_image, heatmaps)
#         self.observed = observed
#         self.priors = priors
#
#
# class GraspFigure(BaseFigureGenerator):
#
#     def __init__(self, rgbd_image, heatmaps, extremas):
#         BaseFigureGenerator.__init__(self, rgbd_image, heatmaps)
#         self.extremas = extremas
#
#         self.grasp_points_img = copy.deepcopy(self.rgbd_image[:, :, 0:3])
#
#     def build(self):
#
#         extrema_dict = dict((e, i)
#                            for i, e in np.ndenumerate(self.extremas)
#                            if e > 0.0)
#
#         sorted_extremas = sorted(extrema_dict, key=lambda key: key, reverse=True)
#
#         for j, extrema in enumerate(sorted_extremas[-5:]):
#
#             max_extrema = extrema_dict[extrema]
#             heat_val = (j * 254 / 5)
#
#             #top
#             for i in range(-5, 5):
#                 self.grasp_points_img[max_extrema[0]-5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#                 self.grasp_points_img[max_extrema[0]-4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#             #bot
#             for i in range(-5, 5):
#                 self.grasp_points_img[max_extrema[0]+4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#                 self.grasp_points_img[max_extrema[0]+5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#             #left
#             for i in range(-5, 5):
#                 self.grasp_points_img[max_extrema[0]+i, max_extrema[1]-5][:3] = [0.0, heat_val, 0.0]
#                 self.grasp_points_img[max_extrema[0]+i, max_extrema[1]-4][:3] = [0.0, heat_val, 0.0]
#             #right
#             for i in range(-5, 5):
#                 self.grasp_points_img[max_extrema[0]+i, max_extrema[1]+5][:3] = [0.0, heat_val, 0.0]
#                 self.grasp_points_img[max_extrema[0]+i, max_extrema[1]+4][:3] = [0.0, heat_val, 0.0]
#
#     def plot(self, figure):
#         plt.title('top 5 grasp points')
#         plt.imshow(self.grasp_points_img)

# def get_local_minima( output):
#     output2 = np.copy(output)
#     e = np.zeros(output2.shape)
#     extrema = argrelextrema(output2, np.less)
#     for i in range(len(extrema[0])):
#         e[extrema[0][i], extrema[1][i]] = output[extrema[0][i], extrema[1][i]]
#
#     return e
#
# def get_local_minima_above_threshold( heatmap):
#     extrema = get_local_minima(heatmap)
#
#     extrema_average = extrema.sum()/(extrema != 0).sum()
#     #threshold is mean of extrema excluding zeros times a scaling factor
#     threshold = extrema_average - .05 * extrema_average
#
#     #set anything negative to 0
#     extrema = np.where(extrema <= threshold, extrema, 0)
#
#     return extrema
#
#
#
# def get_scaled_extremas(rgbd_img, heatmaps, extremas):
#     #extrema imposed on input:
#     border_dim = (2*CROP_BORDER_DIM, 2*CROP_BORDER_DIM)
#     extremas_with_border_shape = [sum(x) for x in zip(extremas.shape, border_dim)] + [3]
#     extremas_with_border = np.zeros(extremas_with_border_shape)
#
#     #extremas_with_border[CROP_BORDER_DIM:-CROP_BORDER_DIM, CROP_BORDER_DIM:-CROP_BORDER_DIM] = extremas[:, :, 0]
#     extremas_with_border[CROP_BORDER_DIM:-CROP_BORDER_DIM, CROP_BORDER_DIM:-CROP_BORDER_DIM] = heatmaps[:, :, :]
#     scaled_extremas = scipy.misc.imresize(extremas_with_border, rgbd_img.shape[0:2], interp='nearest')
#
#     return scaled_extremas


def main():

    dataset_file = paths.choose_from(paths.HEATMAPS_DATASET_DIR)
    dataset = h5py.File(paths.HEATMAPS_DATASET_DIR + dataset_file)

    for i in range(dataset['rgbd_data'].shape[0]):
        rgbd_img = dataset['rgbd_data'][i]
        heatmaps = dataset['heatmaps'][i]
        convolved_heatmaps = dataset['convolved_heatmaps'][i]
        indepent_grasp_points = dataset["independent_grasp_points"][i]
        dependent_grasp_points = dataset["dependent_grasp_points"][i]

        plotter = Plotter()

        plotter.add_subplot('rgb', rgbd_img[:, :, 0:3])
        plotter.add_subplot('d', rgbd_img[:, :, 3])
        plotter.add_subplot('d', rgbd_img[:, :, 3])

        plotter.add_subplot('l_grip_obs', heatmaps[:, :, 0])
        plotter.add_subplot('paml_obs', heatmaps[:, :, 1])
        plotter.add_subplot('r_grip_obs', heatmaps[:, :, 2])

        plotter.add_subplot('l_convolved', convolved_heatmaps[:, :, 0])
        plotter.add_subplot('p_convolved', convolved_heatmaps[:, :, 1])
        plotter.add_subplot('r_convolved', convolved_heatmaps[:, :, 2])

        plotter.add_subplot('l_independent', indepent_grasp_points[0, :, :, :])
        plotter.add_subplot('p_independent', indepent_grasp_points[1, :, :, :])
        plotter.add_subplot('r_independent', indepent_grasp_points[2, :, :, :])

        plotter.add_subplot('l_dependent', dependent_grasp_points[0, :, :, :])
        plotter.add_subplot('p_dependent', dependent_grasp_points[1, :, :, :])
        plotter.add_subplot('r_dependent', dependent_grasp_points[2, :, :, :])


        # extremas = get_local_minima_above_threshold(heatmaps)
        # scaled_extremeas = get_scaled_extremas(rgbd_img, heatmaps, extremas)

        # input_output_figure = InputOutputFigure(rgbd_img, heatmaps)
        # input_output_figure.run()

        # grasps_figure0 = GraspFigure(rgbd_img, heatmaps[:, :, 0], scaled_extremeas[:, :, 0])
        # grasps_figure0.build()
        # plotter.add_subplot('grasp_points0', grasps_figure0.grasp_points_img)
        # #grasps_figure0.run()
        #
        # grasps_figure1 = GraspFigure(rgbd_img, heatmaps[:, :, 1], scaled_extremeas[:, :, 1])
        # grasps_figure1.build()
        # plotter.add_subplot('grasp_points1', grasps_figure1.grasp_points_img)
        # #grasps_figure1.run()
        #
        # grasps_figure2 = GraspFigure(rgbd_img, heatmaps[:, :, 2], scaled_extremeas[:, :, 2])
        # grasps_figure2.build()
        # plotter.add_subplot('grasp_points2', grasps_figure2.grasp_points_img)
        # #grasps_figure2.run()

        #plt.show()
        plotter.show()

if __name__ == '__main__':
    main()





