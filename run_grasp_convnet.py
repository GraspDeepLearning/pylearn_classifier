

import os
import h5py
from scipy.signal import argrelextrema
import scipy.misc
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndenumerate

from grasp_classification_pipeline import (
    np,
    Classification,
    FeatureExtraction,
    Normalization,
    Crop,
    GraspClassificationPipeline)


PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]

PYLEARN_MODEL_DIR = os.path.expanduser('~/grasp_deep_learning/pylearn_classifier_gdl/models/')
PYLEARN_DATASET_DIR = os.path.expanduser('~/grasp_deep_learning/data/')

OUTPUT_DIRECTORY_PATH = os.path.expanduser('~/grasp_deep_learning/data/final_output/')

CROP_BORDER_DIM = 15


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


def drawGrasps(rawImg, extremas, save_filepath, save=True):
    importantExtremas = dict((e, i)
                        for i,e in ndenumerate(extremas) if e > 0.0)
    sortedExtremas = sorted(importantExtremas, key=lambda key: key, reverse=True)
    for key in sortedExtremas:
        print importantExtremas[key]

    import IPython; IPython.embed()
    if save:
        pass
    else:
        pass


def plot(rgbd_img, heatmap, image_index, save=True):

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
    extremas = get_local_minima_above_threshold(heatmap)

    save_filepath = OUTPUT_DIRECTORY_PATH + "output_extremas_" + str(image_index) + ".png"
    plot2d(extremas[:, :, 0], save_filepath, save)

    save_filepath = OUTPUT_DIRECTORY_PATH + "output_3d_extremas_" + str(image_index) + ".png"
    plot3d(extremas, save_filepath, save)

    # extrema imposed on output
    output_with_extremas_imposed = heatmap[:, :, 0] + (extremas[:, :, 0])
    save_filepath = OUTPUT_DIRECTORY_PATH + "output_with_extremas_" + str(image_index) + ".png"
    plot2d(output_with_extremas_imposed, save_filepath, save)

    #extrema imposed on input:
    border_dim = (2*CROP_BORDER_DIM, 2*CROP_BORDER_DIM)
    extremas_with_border_shape = [sum(x) for x in zip(extremas.shape, border_dim)]
    extremas_with_border = np.zeros(extremas_with_border_shape)

    #extremas_with_border[CROP_BORDER_DIM:-CROP_BORDER_DIM, CROP_BORDER_DIM:-CROP_BORDER_DIM] = extremas[:, :, 0]
    extremas_with_border[CROP_BORDER_DIM:-CROP_BORDER_DIM, CROP_BORDER_DIM:-CROP_BORDER_DIM] = heatmap[:, :, 0]
    scaled_extremas = scipy.misc.imresize(extremas_with_border, rgbd_img.shape[0:2])

    input_with_extremas_imposed = np.zeros((480, 640, 3))

    input_with_extremas_imposed[:, :, 0] = np.where(scaled_extremas <= np.percentile(-scaled_extremas, 99), scaled_extremas/scaled_extremas.max(), rgbd_img[:, :, 0])
    input_with_extremas_imposed[:, :, 1] = np.where(scaled_extremas <= np.percentile(-scaled_extremas, 99), 1, rgbd_img[:, :, 1])
    input_with_extremas_imposed[:, :, 2] = np.where(scaled_extremas <= np.percentile(-scaled_extremas, 99), 1, rgbd_img[:, :, 2])

    max_i = np.argmin(scaled_extremas) / 640
    max_j = np.argmin(scaled_extremas) % 640

    input_with_extremas_imposed[max_i-5:max_i+5, max_j-5:max_j+5, :] = 0

    save_filepath = OUTPUT_DIRECTORY_PATH + "input_with_extremas_" + str(image_index) + ".png"
    plot2d(input_with_extremas_imposed, save_filepath, save)


#Lets the user choose an existing pylearn model to run
def get_model():

    models = os.listdir(PYLEARN_MODEL_DIR)

    print
    print "Choose model: "
    print

    for i in range(len(models)):
        print str(i) + ": " + models[i]

    print
    model_index = int(raw_input("Enter Id of model to run (ex 0, 1, or 2): "))

    model = models[model_index]

    return model


#Allows the user to choose a specific dataset to run the model over
def get_dataset_path():

    dataset_folders = os.listdir(PYLEARN_DATASET_DIR)

    print
    print "Choose dataset folder: "
    print

    for i in range(len(dataset_folders)):
        print str(i) + ": " + dataset_folders[i]

    print
    dataset_folder_index = int(raw_input("Enter Id of dataset folder (ex 0, 1, or 2): "))
    dataset_folder = dataset_folders[dataset_folder_index]


    datasets = os.listdir(PYLEARN_DATASET_DIR + dataset_folder + "/")

    print
    print "Choose dataset folder: "
    print

    for i in range(len(datasets)):
        print str(i) + ": " + datasets[i]

    print
    dataset_index = int(raw_input("Enter Id of dataset folder (ex 0, 1, or 2): "))
    dataset = datasets[dataset_index]

    return (PYLEARN_DATASET_DIR + dataset_folder + "/" + dataset)


if __name__ == "__main__":

    conv_model_name = get_model()
    conv_model_filepath = PYLEARN_MODEL_DIR + conv_model_name + "/cnn_model.pkl"

    grasp_classification_pipeline = GraspClassificationPipeline(conv_model_filepath, border_dim=CROP_BORDER_DIM, useFloat64=True)

    dataset_filepath = get_dataset_path()
    dataset = h5py.File(dataset_filepath)

    rgbd_images = dataset['rgbd_data']
    num_images = rgbd_images.shape[0]

    for image_index in range(num_images):

        print str(image_index) + "/" + str(num_images)

        rgbd_img = rgbd_images[image_index]

        start = time.time()
        heatmap = grasp_classification_pipeline.run(rgbd_img)
        print time.time() - start

        plot(rgbd_img, heatmap, image_index, save=False)



