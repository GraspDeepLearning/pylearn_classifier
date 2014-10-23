
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
from subtractive_divisive_lcn import *


class CopyInRaw():

    def __init__(self, raw_rgbd_dataset_filepath, raw_data_in_key='images', raw_data_out_key='rgbd_data'):
        self.raw_rgbd_dataset = h5py.File(raw_rgbd_dataset_filepath)
        self.raw_data_in_key = raw_data_in_key
        self.raw_data_out_key = raw_data_out_key

    def run(self, dataset, index):
        dataset[self.raw_data_out_key][index] = self.raw_rgbd_dataset[self.raw_data_in_key][index]


class LecunSubtractiveDivisiveLCN():

    def __init__(self, kernel_shape=9, in_key='rgbd_data', out_key='rgbd_data_normalized'):
        self.kernel_shape = kernel_shape
        self.in_key = in_key
        self.out_key = out_key
        self.sub_div_lcn = None

    def run(self, dataset, index):
        img = dataset[self.in_key][index]
        num_channels = img.shape[2]

        img_out = np.zeros_like(img)
        img_in = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32)

        if not self.sub_div_lcn:
            self.sub_div_lcn = subtractive_divisive_lcn(img_in, img_shape=img.shape[0:2], kernel_shape=self.kernel_shape)

        for i in range(num_channels):
            img_in[0] = img[:, :, i]
            img_out[:, :, i] = self.sub_div_lcn(img_in.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], 1)))

        dataset[self.out_key][index] = img_out


class FeatureExtraction():

    def __init__(self, model_filepath,
                 in_key='rgbd_data_normalized',
                 out_key='extracted_features',
                 use_float_64=False,
                 shape=(1024, 1280),
                 num_channels=1):

        f = open(model_filepath)
        cnn_model = cPickle.load(f)

        self.shape = shape
        self.in_key = in_key
        self.out_key = out_key

        if use_float_64:
            self.float_type_str = 'float64'
            self.float_dtype = np.float64

        else:
            self.float_type_str = 'float32'
            self.float_dtype = np.float32



        new_space = pylearn2.space.Conv2DSpace(shape, num_channels=num_channels, axes=('c', 0, 1, 'b'), dtype=self.float_type_str)

        start_classifier_index = 0
        for i in range(len(cnn_model.layers)):
            if not isinstance(cnn_model.layers[i], pylearn2.models.mlp.ConvRectifiedLinear):
                start_classifier_index = i
                break

        classifier_layers = cnn_model.layers[start_classifier_index:]
        cnn_model.layers = cnn_model.layers[0:start_classifier_index]


        weights = []
        biases = []

        for layer in  cnn_model.layers:
            weights.append(np.copy(layer.get_weights_topo()))
            biases.append(np.copy(layer.get_biases()))

        #import IPython
        #IPython.embed()

        cnn_model.set_batch_size(1)
        cnn_model.set_input_space(new_space)

        # for i in range(len(cnn_model.layers)):
        #     weights_rolled = np.rollaxis(weights[i], 3, 1)
        #     cnn_model.layers[i].set_weights(weights_rolled)
        #     cnn_model.layers[i].set_biases(biases[i])

        X = cnn_model.get_input_space().make_theano_batch()
        Y = cnn_model.fprop(X)

        self._feature_extractor = theano.function([X], Y)


    def run(self, dataset, index):

        img_in = dataset[self.in_key][index]
        num_channels = img_in.shape[2]

        img = np.zeros((num_channels, self.shape[0], self.shape[1], 1), dtype=self.float_dtype)

        img[:, :, :, 0] = np.rollaxis(img_in, 2, 0)

        out_raw = self._feature_extractor(img)

        out_rolled = np.rollaxis(out_raw, 1, 4)
        out_window = out_rolled[0, :, :, :]

        dataset[self.out_key][index] = out_window


class Classification():

    def __init__(self, model_filepath):

        f = open(model_filepath)

        cnn_model = cPickle.load(f)

        start_classifier_index = 0
        for i in range(len(cnn_model.layers)):
            if not isinstance(cnn_model.layers[i], pylearn2.models.mlp.ConvRectifiedLinear):
                start_classifier_index = i
                break

        layers = cnn_model.layers[start_classifier_index:]

        self.Ws = []
        #self.bs = []

        self.Ws.append(layers[0].get_weights_topo())

        for i in range(len(layers)):
            if i != 0:
                layer = layers[i]
                print type(layer)
                self.Ws.append(layer.get_weights())
                #self.bs.append(layer.get_biases())

    def run(self, dataset, index):

        X = dataset['extracted_features'][index]

        W0 = self.Ws[0]
        out = np.dot(X, W0)[:, :, :, 0, 0]

        for i in range(len(self.Ws)):
            if i != 0:
                out = np.dot(out, self.Ws[i])


        dataset['heatmaps'][index] = out


class HeatmapNormalization():

    def __init__(self, in_key='heatmaps', out_key='normalized_heatmaps'):
        self.in_key = in_key
        self.out_key = out_key
        self.max = 255.0

    def run(self, dataset, index):
        #currently heatmaps.min() is < 0 and heatmaps.max() > 0
        #normalized between 0 and 255
        heatmaps = dataset[self.in_key][index]

        normalize_heatmaps = self.max-(heatmaps-heatmaps.min())/(heatmaps.max()-heatmaps.min())*self.max
        dataset[self.out_key][index] = normalize_heatmaps


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

        heatmaps = dataset['normalized_heatmaps'][index]
        img_in_shape = (416, 576)
        l_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 0], img_in_shape)
        palm_obs = scipy.misc.imresize(heatmaps[:, :, 1], img_in_shape)
        r_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 2], img_in_shape)

        img_in = np.zeros((1, 1, img_in_shape[0], img_in_shape[1]), dtype=np.float32)

        img_in[:, :] = l_gripper_obs
        l_gripper_conv = self.f(img_in)
        dataset["l_convolved_heatmaps"][index] = l_gripper_conv

        img_in[:, :] = r_gripper_obs
        r_gripper_conv = self.f(img_in)
        dataset["r_convolved_heatmaps"][index] = r_gripper_conv

        img_in[:, :] = palm_obs
        palm_conv = self.f(img_in)
        dataset["p_convolved_heatmaps"][index] = palm_conv

        out_shape = palm_conv[0, 0].shape
        x_border = (img_in_shape[0]-out_shape[0])/2
        y_border = (img_in_shape[1]-out_shape[1])/2
        l_gripper_obs2 = l_gripper_obs[x_border:-x_border - 1, y_border:-y_border - 1]
        palm_obs2 = palm_obs[x_border:-x_border - 1, y_border:-y_border - 1]
        r_gripper_obs2 = r_gripper_obs[x_border:-x_border - 1, y_border:-y_border - 1]

        l_gripper_out = l_gripper_obs2 * palm_conv[0, 0] * r_gripper_conv[0, 1]
        palm_out = palm_obs2 * l_gripper_conv[0, 2] * r_gripper_conv[0, 3]
        r_gripper_out = r_gripper_obs2 * l_gripper_conv[0, 4] * palm_conv[0, 5]

        dataset['convolved_heatmaps'][index, :, :, 0] = l_gripper_out
        dataset['convolved_heatmaps'][index, :, :, 1] = palm_out
        dataset['convolved_heatmaps'][index, :, :, 2] = r_gripper_out


class CalculateMax():


    def run(self, dataset, index):

        out_shape = dataset['convolved_heatmaps'][index][:, :, 0].shape
        l_gripper_out = dataset['convolved_heatmaps'][index][:, :, 0]
        palm_out = dataset['convolved_heatmaps'][index][:, :, 1]
        r_gripper_out = dataset['convolved_heatmaps'][index][:, :, 2]
        rgb_with_grasp = dataset["best_grasp"][index]

        img_in_shape = dataset["rgbd_data"][index, :, :, 0].shape
        x_border = (img_in_shape[0]-out_shape[0])/2
        y_border = (img_in_shape[1]-out_shape[1])/2

        rgb_with_grasp[:] = np.copy(dataset["rgbd_data"][index, x_border:-x_border - 1, y_border:-y_border - 1, 0:3])

        l_max = np.argmax(l_gripper_out)
        p_max = np.argmax(palm_out)
        r_max = np.argmax(r_gripper_out)

        lim = out_shape[1]
        l_max_x, l_max_y = (l_max / lim, l_max % lim)
        p_max_x, p_max_y = (p_max / lim, p_max % lim)
        r_max_x, r_max_y = (r_max / lim, r_max % lim)

        try:
            rgb_with_grasp[l_max_x-5:l_max_x + 5, l_max_y-5:l_max_y + 5] = [0, 0, 0]
            rgb_with_grasp[p_max_x-5:p_max_x + 5, p_max_y-5:p_max_y + 5] = [0, 0, 0]
            rgb_with_grasp[r_max_x-5:r_max_x + 5, r_max_y-5:r_max_y + 5] = [0, 0, 0]
        except:
            pass

        dataset['best_grasp'][index] = rgb_with_grasp






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
        extrema = argrelextrema(output2, np.greater)
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
            local_minima = self.get_local_minima(heatmap)
            scaled_extremas = self.get_scaled_extremas(rgbd_img, heatmap, local_minima)

            extrema_dict = dict((e, i)
                               for i, e in np.ndenumerate(scaled_extremas)
                               if e > 0.0)

            sorted_extremas = sorted(extrema_dict, key=lambda key: key, reverse=True)

            for j, extrema in enumerate(sorted_extremas[-20:]):

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
        h5py_file = h5py.File(in_filepath)
        if 'rgbd_data' in h5py_file.keys():
            self._num_images = h5py_file['rgbd_data'].shape[0]
        else:
            self._num_images = h5py_file['image'].shape[0]
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








