

import os
import h5py
import collections
import pylab
from scipy.signal import argrelextrema
import scipy.misc
import time

import pylearn2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from grasp_classification_pipeline import *

PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]

CONV_MODEL_FILEPATH = 'models/grasp_72x72_model/cnn_model.pkl'
DATASET_FILEPATH = '/home/jvarley/grasp_deep_learning/data/rgbd_images/saxena_partial_rgbd_and_labels.h5'
OUTPUT_DIRECTORY_PATH = "/home/jvarley/grasp_deep_learning/data/final_output/"


def embedIPython():
    import IPython
    IPython.embed()
    assert False


def get_scaled_image(img, scale=(54, 74)):
    return scipy.misc.imresize(img, scale)


def get_local_extrema(output):
    output2 = np.copy(output)*-1
    e = np.zeros(output2.shape)
    extrema = argrelextrema(output2, np.greater)
    for i in range(len(extrema[0])):
        e[extrema[0][i], extrema[1][i], extrema[2][i]] = output[extrema[0][i], extrema[1][i], extrema[2][i]]

    return e


def get_positive_extrema(heatmap):
    extrema = get_local_extrema(heatmap)

    #threshold is mean of extrema excluding zeros times a scaling factor
    threshold = extrema.sum()/(extrema != 0).sum()

    #subtract threshold
    extrema = extrema - threshold

    #set anything negative to 0
    extrema = np.where(extrema >= 0, extrema, 0)

    return extrema


def plot3d(img, save_filepath, save=True):

    plt.figure()
    ax = plt.axes(projection='3d')

    x = np.zeros(img.shape[0]*img.shape[1])
    y = np.zeros(img.shape[0]*img.shape[1])
    z = np.zeros(img.shape[0]*img.shape[1])

    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x[count] = i
            y[count] = j
            z[count] = img[i, j, 0]
            count += 1

    ax.scatter(x, y, z)

    if save:
        plt.savefig(save_filepath)
    else:
        plt.title(save_filepath)
        plt.show()
        plt.close()


def plot2d(img, save_filepath, save=True):

    if save:
        scipy.misc.imsave(save_filepath, img)
    else:
        plt.imshow(img)
        plt.title(save_filepath)
        plt.show()
        plt.close()


def plot(rgbd_img, heatmap, image_index,  save=True):

    #raw_input
    save_filepath = OUTPUT_DIRECTORY_PATH + "rgb_input_" + str(image_index) + ".png"
    plot2d(rgbd_img[:, :, 0:3], save_filepath, save)

    #3d scatter_plot
    save_filepath = OUTPUT_DIRECTORY_PATH + "output_3D_" + str(image_index) + ".png"
    plot3d(heatmap, save_filepath, save)

    #show output heat map
    save_filepath = OUTPUT_DIRECTORY_PATH + "output_heatmap_" + str(image_index) + ".png"
    plot2d(heatmap[:, :, 0], save_filepath, save)

    # show extremas
    extremas = get_positive_extrema(heatmap)
    save_filepath = OUTPUT_DIRECTORY_PATH + "output_extremas_" + str(image_index) + ".png"
    plot2d(extremas[:, :, 0], save_filepath, save)

    save_filepath = OUTPUT_DIRECTORY_PATH + "output_3d_extremas_" + str(image_index) + ".png"
    plot3d(extremas, save_filepath, save)

    # extrema imposed on output
    output_with_extremas_imposed = heatmap[:, :, 0] + (extremas[:, :, 0])
    save_filepath = OUTPUT_DIRECTORY_PATH + "output_with_extremas_" + str(image_index) + ".png"
    plot2d(output_with_extremas_imposed, save_filepath, save)

    #extrema imposed on input:
    scaled_img = get_scaled_image(rgbd_img, extremas.shape[0:2])[:,:,0]
    input_with_extremas_imposed = (scaled_img + extremas[:, :, 0])/(scaled_img + extremas[:, :, 0]).max()
    save_filepath = OUTPUT_DIRECTORY_PATH + "input_with_extremas_" + str(image_index) + ".png"
    plot2d(input_with_extremas_imposed, save_filepath, save)


if __name__ == "__main__":

    grasp_classification_pipeline = GraspClassificationPipeline(CONV_MODEL_FILEPATH)

    dataset = h5py.File(DATASET_FILEPATH)

    rgbd_images = dataset['rgbd_data']
    num_images = rgbd_images.shape[0]

    for image_index in range(num_images):

        print str(image_index) + "/" + str(num_images)

        rgbd_img = rgbd_images[image_index]
        heatmap = grasp_classification_pipeline.run(rgbd_img)
        plot(rgbd_img, heatmap, image_index, save=True)



