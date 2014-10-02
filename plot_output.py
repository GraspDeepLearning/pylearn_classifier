
import os
import h5py
from scipy.signal import argrelextrema
import scipy.misc
import copy
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import paths

CROP_BORDER_DIM = 15


class BaseFigureGenerator():

    _figure_number = 0

    def __init__(self, rgbd_image, heatmaps):
        self.rgbd_image = rgbd_image
        self.heatmaps = heatmaps

    @staticmethod
    def get_figure_number():
        BaseFigureGenerator._figure_number += 1
        return BaseFigureGenerator._figure_number

    def build(self):
        pass

    def plot(self):
        pass

    def run(self):
        self.build()
        figure = plt.figure(BaseFigureGenerator.get_figure_number())
        self.plot(figure)


class InputOutputFigure(BaseFigureGenerator):

    def plot(self,figure):

        subplot_index = 1

        plt.subplot(3, 3, subplot_index)
        subplot_index += 1
        plt.title('rgb')
        plt.imshow(self.rgbd_image[:, :, 0:3])

        plt.subplot(3, 3, subplot_index)
        subplot_index += 2
        plt.title('d')
        plt.imshow(self.rgbd_image[:, :, 3])

        for i in range(self.heatmaps.shape[2]):
            plt.subplot(3, 3, subplot_index)
            subplot_index += 1
            plt.title('heatmap_' + str(i))
            plt.imshow(self.heatmaps[:, :, i])

        for index in range(self.heatmaps.shape[2]):
            heatmap = self.heatmaps[:,:,index]

            #ax = plt.axes(projection='3d')
            ax = figure.add_subplot(3, 3, subplot_index, projection='3d')
            subplot_index += 1
            plt.title('heatmap_' + str(index))

            x = np.zeros(heatmap.shape[0]*heatmap.shape[1])
            y = np.zeros(heatmap.shape[0]*heatmap.shape[1])
            z = np.zeros(heatmap.shape[0]*heatmap.shape[1])

            count = 0
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[1]):
                    x[count] = i
                    y[count] = j
                    z[count] = heatmap[i, j]
                    count += 1

            ax.scatter(x, y, z)


class GraspFigure(BaseFigureGenerator):

    def __init__(self, rgbd_image, heatmaps, extremas):
        BaseFigureGenerator.__init__(self, rgbd_image, heatmaps)
        self.extremas = extremas

        self.grasp_points_img = copy.deepcopy(self.rgbd_image[:, :, 0:3])

    def build(self):

        extrema_dict = dict((e, i)
                           for i, e in np.ndenumerate(self.extremas)
                           if e > 0.0)

        sorted_extremas = sorted(extrema_dict, key=lambda key: key, reverse=True)

        for j, extrema in enumerate(sorted_extremas[-5:]):

            max_extrema = extrema_dict[extrema]
            heat_val = (j * 254 / 5)

            #top
            for i in range(-5, 5):
                self.grasp_points_img[max_extrema[0]-5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
                self.grasp_points_img[max_extrema[0]-4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
            #bot
            for i in range(-5, 5):
                self.grasp_points_img[max_extrema[0]+4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
                self.grasp_points_img[max_extrema[0]+5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
            #left
            for i in range(-5, 5):
                self.grasp_points_img[max_extrema[0]+i, max_extrema[1]-5][:3] = [0.0, heat_val, 0.0]
                self.grasp_points_img[max_extrema[0]+i, max_extrema[1]-4][:3] = [0.0, heat_val, 0.0]
            #right
            for i in range(-5, 5):
                self.grasp_points_img[max_extrema[0]+i, max_extrema[1]+5][:3] = [0.0, heat_val, 0.0]
                self.grasp_points_img[max_extrema[0]+i, max_extrema[1]+4][:3] = [0.0, heat_val, 0.0]

    def plot(self, figure):
        plt.title('top 5 grasp points')
        plt.imshow(self.grasp_points_img)


def get_local_minima(output):
    output2 = np.copy(output)
    e = np.zeros(output2.shape)
    extrema = argrelextrema(output2, np.less)
    for i in range(len(extrema[0])):
        e[extrema[0][i], extrema[1][i], extrema[2][i]] = output[extrema[0][i], extrema[1][i], extrema[2][i]]

    return e


def get_local_minima_above_threshold(heatmap):
    extrema = get_local_minima(heatmap)

    extrema_average =  extrema.sum()/(extrema != 0).sum()
    #threshold is mean of extrema excluding zeros times a scaling factor
    threshold = extrema_average - .05 * extrema_average

    #set anything negative to 0
    extrema = np.where(extrema <= threshold, extrema, 0)

    return extrema


def get_scaled_extremas(rgbd_img, heatmaps, extremas):
    #extrema imposed on input:
    border_dim = (2*CROP_BORDER_DIM, 2*CROP_BORDER_DIM)
    extremas_with_border_shape = [sum(x) for x in zip(extremas.shape, border_dim)]
    extremas_with_border = np.zeros(extremas_with_border_shape)

    #extremas_with_border[CROP_BORDER_DIM:-CROP_BORDER_DIM, CROP_BORDER_DIM:-CROP_BORDER_DIM] = extremas[:, :, 0]
    extremas_with_border[CROP_BORDER_DIM:-CROP_BORDER_DIM, CROP_BORDER_DIM:-CROP_BORDER_DIM] = heatmaps[:, :, 0]
    scaled_extremas = scipy.misc.imresize(extremas_with_border, rgbd_img.shape[0:2], interp='nearest')

    return scaled_extremas


#Allows the user to choose a specific dataset to run the model over
def get_dataset_file():

    datasets = os.listdir(paths.HEATMAPS_DATASET_DIR)

    print
    print "Choose dataset file: "
    print

    for i in range(len(datasets)):
        print str(i) + ": " + datasets[i]

    print
    dataset_index = int(raw_input("Enter Id of dataset file (ex 0, 1, or 2): "))
    dataset_file = datasets[dataset_index]

    return h5py.File(paths.HEATMAPS_DATASET_DIR + dataset_file)


def main():
    dataset = get_dataset_file()

    for i in range(dataset['rgbd_data'].shape[0]):
        rgbd_img = dataset['rgbd_data'][i]
        heatmaps = dataset['heatmaps'][i]

        extremas = get_local_minima_above_threshold(heatmaps)
        scaled_extremeas = get_scaled_extremas(rgbd_img, heatmaps, extremas)

        input_output_figure = InputOutputFigure(rgbd_img, heatmaps)
        input_output_figure.run()

        grasps_figure = GraspFigure(rgbd_img, heatmaps, scaled_extremeas)
        grasps_figure.run()

        plt.show()

if __name__ == '__main__':
    main()





