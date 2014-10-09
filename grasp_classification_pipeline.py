
import theano
import pylearn2
import cPickle
import copy
import numpy as np
import h5py
import time
import scipy.signal
import scipy.misc
from scipy.signal import argrelextrema
from theano.tensor.nnet import conv

import matplotlib.pyplot as plt


class CopyInRaw():

    def __init__(self, raw_rgbd_dataset_filepath):
        self.raw_rgbd_dataset = h5py.File(raw_rgbd_dataset_filepath)

    def run(self, dataset, index):
        dataset['rgbd_data'][index] = self.raw_rgbd_dataset['rgbd_data'][index]


class NormalizeRaw():

    def __init__(self):
        pass

    def run(self,dataset,index):
        rgbd_img = dataset['rgbd_data'][index]

        rgbd_img_norm = np.zeros_like(rgbd_img)
        for i in range(4):
            rgbd_img_norm[:, :, i] = (rgbd_img[:, :, i] + rgbd_img[:, :, i].min()) / (rgbd_img[:, :, i].max() + rgbd_img[:, :, i].min())

        dataset['rgbd_data_normalized'][index] = rgbd_img_norm


class FeatureExtraction():

    def __init__(self, model_filepath, useFloat64=False):

        f = open(model_filepath)
        cnn_model = cPickle.load(f)
        self.useFloat64 = useFloat64

        if self.useFloat64:
            new_space = pylearn2.space.Conv2DSpace((480, 640), num_channels=4, axes=('c', 0, 1, 'b'), dtype='float64')
        else:
            new_space = pylearn2.space.Conv2DSpace((480, 640), num_channels=4, axes=('c', 0, 1, 'b'), dtype='float32')

        cnn_model.layers = cnn_model.layers[0:-1]

        #we want to padd zeros around the edges rather than ignoring edge pixels
        for i in range(len(cnn_model.layers)):
            cnn_model.layers[i].border_mode = "full"

        cnn_model.set_batch_size(1)
        cnn_model.set_input_space(new_space)

        X = cnn_model.get_input_space().make_theano_batch()
        Y = cnn_model.fprop(X)

        self._feature_extractor = theano.function([X], Y)

    def run(self, dataset, index):

        img_in = dataset['rgbd_data_normalized'][index]

        if self.useFloat64:
            img = np.zeros((4, 480, 640, 1), dtype=np.float64)
        else:
            img = np.zeros((4, 480, 640, 1), dtype=np.float32)

        img[:, :, :, 0] = np.rollaxis(img_in, 2, 0)

        out_raw = self._feature_extractor(img)
        out_rolled = np.rollaxis(out_raw, 1, 4)
        out_window = out_rolled[0, :, :, :]

        dataset['extracted_features'][index] = out_window


class Classification():

    def __init__(self, model_filepath):

        f = open(model_filepath)

        cnn_model = cPickle.load(f)
        cnn_model = cnn_model.layers[-1]

        W = cnn_model.get_weights_topo()
        #W = W[0, 0, :, :]

        b = cnn_model.b.get_value()

        self.W = W
        self.b = b

    def run(self, dataset, index):
        X = dataset['extracted_features'][index]

        heatmaps = np.dot(X, self.W)[:, :, :, 0, 0] + self.b

        dataset['heatmaps'][index] = heatmaps


class Normalization():

    def __init__(self):
        self.max = 255.0

    def run(self, dataset, index):
        #currently heatmaps.min() is < 0 and heatmaps.max() > 0
        #normalized between 0 and 255
        heatmaps = dataset['heatmaps'][index]

        normalize_heatmaps = self.max-(heatmaps-heatmaps.min())/(heatmaps.max()-heatmaps.min())*self.max
        dataset['normalized_heatmaps'][index] = normalize_heatmaps


class Crop():

    def __init__(self, border_dim=15):
        self.border_dim = border_dim

    def run(self, dataset, index):
        heatmaps = dataset['normalized_heatmaps'][index]
        cropped_heatmaps = heatmaps[self.border_dim:-self.border_dim, self.border_dim:-self.border_dim]
        dataset['cropped_heatmaps'][index] = cropped_heatmaps


class ConvolvePriors():

    def __init__(self, priors_filepath):
        self.priors = h5py.File(priors_filepath)
        l_g_p = self.priors['l_gripper_given_palm_blur_norm'][100:200, 100:200]
        l_g_r = self.priors['l_gripper_given_r_gripper_blur_norm'][100:200, 100:200]

        p_g_l = self.priors['palm_given_l_gripper_blur_norm'][100:200, 100:200]
        p_g_r = self.priors['palm_given_r_gripper_blur_norm'][100:200, 100:200]

        r_g_l = self.priors['r_gripper_given_l_gripper_blur_norm'][100:200, 100:200]
        r_g_p = self.priors['r_gripper_given_palm_blur_norm'][100:200, 100:200]

        input = theano.tensor.tensor4(name='input')
        w_shape = (6, 1, 100, 100)
        w = np.zeros(w_shape)
        w[0, 0] = l_g_p
        w[1, 0] = l_g_r

        w[2, 0] = p_g_l
        w[3, 0] = p_g_r

        w[4, 0] = r_g_l
        w[5, 0] = r_g_p

        W = theano.shared(np.asarray(w, dtype=input.dtype), name='W')
        self.conv_out = conv.conv2d(input, W)
        self.f = theano.function([input], self.conv_out)



    def run(self, dataset, index):

        heatmaps = dataset['cropped_heatmaps'][index]

        l_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 0], (480, 640))
        palm_obs = scipy.misc.imresize(heatmaps[:, :, 1], (480, 640))
        r_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 2], (480, 640))

        img_in = np.zeros((1, 1, 480, 640), dtype=np.float32)

        img_in[:, :] = l_gripper_obs
        out = self.f(img_in)
        l_gripper_out = out[0, 0] * out[0, 1]

        img_in[:, :] = r_gripper_obs
        out = self.f(img_in)
        r_gripper_out = out[0, 4] * out[0, 5]

        img_in[:, :] = palm_obs
        out = self.f(img_in)
        palm_out = out[0, 2] * out[0, 3]

        out = np.zeros((381, 541, 3))
        out[:, :, 0] = l_gripper_out
        out[:, :, 1] = palm_out
        out[:, :, 2] = r_gripper_out

        dataset['convolved_heatmaps'][index] = out


class CalculateTopFive():

    def __init__(self, input_key='convolved_heatmaps',
                 output_key='dependent_grasp_points',
                 border_dim=15):
        self.input_key = input_key
        self.output_key = output_key
        self.border_dim = border_dim

    def get_local_minima(self, output):
        output2 = np.copy(output)
        e = np.zeros(output2.shape)
        extrema = argrelextrema(output2, np.less)
        for i in range(len(extrema[0])):
            e[extrema[0][i], extrema[1][i]] = output[extrema[0][i], extrema[1][i]]

        return e

    def get_local_minima_above_threshold(self, heatmap):
        extrema = self.get_local_minima(heatmap)

        extrema_average = extrema.sum()/(extrema != 0).sum()
        #threshold is mean of extrema excluding zeros times a scaling factor
        threshold = extrema_average - .05 * extrema_average

        #set anything negative to 0
        extrema = np.where(extrema <= threshold, extrema, 0)

        return extrema

    def get_scaled_extremas(self, rgbd_img, heatmaps, extremas):
        #extrema imposed on input:
        border_dim = (2*self.border_dim, 2*self.border_dim)
        extremas_with_border_shape = [sum(x) for x in zip(extremas.shape, border_dim)]
        extremas_with_border = np.zeros(extremas_with_border_shape)

        extremas_with_border[self.border_dim:-self.border_dim, self.border_dim:-self.border_dim] = heatmaps[:, :]
        scaled_extremas = scipy.misc.imresize(extremas_with_border, rgbd_img.shape[0:2], interp='nearest')

        return scaled_extremas

    def run(self, dataset, index):

        heatmaps = dataset[self.input_key][index]
        rgbd_img = dataset['rgbd_data'][index]

        grasp_points_img = copy.deepcopy(rgbd_img[:, :, 0:3])

        for heatmap_index in range(3):
            heatmap = heatmaps[:, :, heatmap_index]


            local_minima = self.get_local_minima_above_threshold(heatmap)
            scaled_extremas = self.get_scaled_extremas(rgbd_img, heatmap, local_minima)

            extrema_dict = dict((e, i)
                               for i, e in np.ndenumerate(scaled_extremas)
                               if e > 0.0)

            sorted_extremas = sorted(extrema_dict, key=lambda key: key, reverse=True)

            for j, extrema in enumerate(sorted_extremas[-5:]):

                max_extrema = extrema_dict[extrema]
                heat_val = (j * 254 / 5)

                #top
                for i in range(-5, 5):
                    grasp_points_img[max_extrema[0]-5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
                    grasp_points_img[max_extrema[0]-4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
                #bot
                for i in range(-5, 5):
                    grasp_points_img[max_extrema[0]+4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
                    grasp_points_img[max_extrema[0]+5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
                #left
                for i in range(-5, 5):
                    grasp_points_img[max_extrema[0]+i, max_extrema[1]-5][:3] = [0.0, heat_val, 0.0]
                    grasp_points_img[max_extrema[0]+i, max_extrema[1]-4][:3] = [0.0, heat_val, 0.0]
                #right
                for i in range(-5, 5):
                    grasp_points_img[max_extrema[0]+i, max_extrema[1]+5][:3] = [0.0, heat_val, 0.0]
                    grasp_points_img[max_extrema[0]+i, max_extrema[1]+4][:3] = [0.0, heat_val, 0.0]

            dataset[self.output_key][index, heatmap_index] = grasp_points_img


class GraspClassificationPipeline():

    def __init__(self, out_filepath, in_filepath):

        self.dataset = h5py.File(out_filepath)
        self._num_images = h5py.File(in_filepath)['rgbd_data'].shape[0]
        self._pipeline_stages = []

    def add_stage(self, stage):
        self._pipeline_stages.append(stage)

    def run(self):

        for index in range(self._num_images):

            print
            print 'starting ' + str(index) + " of " + str(self._num_images)
            print

            for stage in self._pipeline_stages:
                start_time = time.time()
                stage.run(self.dataset, index)
                print str(stage) + ' took ' + str(time.time() - start_time) + ' seconds to complete.'








