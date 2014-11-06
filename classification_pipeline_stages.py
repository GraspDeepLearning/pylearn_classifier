
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

import cPickle
import pylearn2.models.mlp


class ClassificationStage():

    def __init__(self, in_key, out_key):
        self.in_key = in_key
        self.out_key = out_key

    def dataset_inited(self, dataset):
        return self.out_key in dataset.keys()

    def init_dataset(self, dataset):
        out = self._run(dataset, 0)

        shape = (900, out.shape[0], out.shape[1], out.shape[2])
        chunk_size = (10, out.shape[0], out.shape[1], out.shape[2])

        dataset.create_dataset(self.out_key, shape, chunks=chunk_size)

    def run(self, dataset, index):
        out = self._run(dataset, index)
        dataset[self.out_key][index] = out

    def _run(self, dataset, index):
        print "Base class _run should not be called."
        raise NotImplementedError


class CopyInRaw(ClassificationStage):

    def __init__(self, raw_rgbd_dataset_filepath, in_key='images', out_key='rgbd_data'):
        self.raw_rgbd_dataset = h5py.File(raw_rgbd_dataset_filepath)
        self.in_key = in_key
        self.out_key = out_key

    def init_dataset(self, dataset):
        shape = self.raw_rgbd_dataset[self.in_key].shape
        chunk_size = (10, shape[1], shape[2], shape[3])
        dataset.create_dataset(self.out_key, shape, chunks=chunk_size)

    def run(self, dataset, index):
        dataset[self.out_key][index] = self.raw_rgbd_dataset[self.in_key][index]


class LecunSubtractiveDivisiveLCN(ClassificationStage):

    def __init__(self, kernel_shape=9, in_key='rgbd_data', out_key='rgbd_data_normalized'):
        self.kernel_shape = kernel_shape
        self.in_key = in_key
        self.out_key = out_key
        self.sub_div_lcn = None

    def _run(self, dataset, index):
        img = dataset[self.in_key][index]
        num_channels = img.shape[2]

        img_out = np.zeros_like(img)
        img_in = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32)

        if not self.sub_div_lcn:
            self.sub_div_lcn = subtractive_divisive_lcn(img_in, img_shape=img.shape[0:2], kernel_shape=self.kernel_shape)

        for i in range(num_channels):
            img_in[0] = img[:, :, i]
            img_out[:, :, i] = self.sub_div_lcn(img_in.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], 1)))

        return img_out


class FeatureExtraction(ClassificationStage):

    def __init__(self, model_filepath,
                 in_key='rgbd_data_normalized',
                 out_key='extracted_features',
                 use_float_64=False):

        ClassificationStage.__init__(self, in_key, out_key)

        self.model_filepath = model_filepath

        if use_float_64:
            self.float_type_str = 'float64'
            self.float_dtype = np.float64

        else:
            self.float_type_str = 'float32'
            self.float_dtype = np.float32

        # this is defined in init_dataset
        # we need to know the size of the input
        # in order to init this
        self._feature_extractor = None
        self.shape = None
        self.num_channels = None

    def init_dataset(self, dataset):

        f = open(self.model_filepath)

        cnn_model = cPickle.load(f)

        img_in = dataset[self.in_key][0]
        self.shape = img_in.shape[0:2]
        self.num_channels = img_in.shape[-1]

        new_space = pylearn2.space.Conv2DSpace(self.shape, num_channels=self.num_channels, axes=('c', 0, 1, 'b'), dtype=self.float_type_str)

        start_classifier_index = 0
        for i in range(len(cnn_model.layers)):
            if not isinstance(cnn_model.layers[i], pylearn2.models.mlp.ConvRectifiedLinear):
                start_classifier_index = i
                break

        cnn_model.layers = cnn_model.layers[0:start_classifier_index]

        weights = []
        biases = []

        for layer in cnn_model.layers:
            weights.append(np.copy(layer.get_weights_topo()))
            biases.append(np.copy(layer.get_biases()))

        cnn_model.set_batch_size(1)
        cnn_model.set_input_space(new_space)

        for i in range(len(cnn_model.layers)):
            weights_rolled = np.rollaxis(weights[i], 3, 1)
            cnn_model.layers[i].set_weights(weights_rolled)
            cnn_model.layers[i].set_biases(biases[i])

        X = cnn_model.get_input_space().make_theano_batch()
        Y = cnn_model.fprop(X)

        self._feature_extractor = theano.function([X], Y)

        ClassificationStage.init_dataset(self, dataset)

    def _run(self, dataset, index):
        img_in = dataset[self.in_key][index]

        img = np.zeros((self.num_channels, self.shape[0], self.shape[1], 1), dtype=self.float_dtype)

        img[:, :, :, 0] = np.rollaxis(img_in, 2, 0)

        out_raw = self._feature_extractor(img)

        out_rolled = np.rollaxis(out_raw, 1, 4)
        out_window = out_rolled[0, :, :, :]

        return out_window


class Classification(ClassificationStage):

    def __init__(self, model_filepath, in_key='extracted_features', out_key='heatmaps' ):

        self.in_key = in_key
        self.out_key = out_key

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

    def _run(self, dataset, index):

        X = dataset[self.in_key][index]

        W0 = self.Ws[0]
        out = np.dot(X, W0)[:, :, :, 0, 0]

        for i in range(len(self.Ws)):
            if i != 0:
                out = np.dot(out, self.Ws[i])

        return out


class HeatmapNormalization(ClassificationStage):

    def __init__(self, in_key='heatmaps', out_key='normalized_heatmaps'):
        self.in_key = in_key
        self.out_key = out_key
        self.max = 255.0

    def _run(self, dataset, index):
        heatmaps = dataset[self.in_key][index]
        normalize_heatmaps = self.max-(heatmaps-heatmaps.min())/(heatmaps.max()-heatmaps.min())*self.max
        return normalize_heatmaps





class Rescale(ClassificationStage):

    def __init__(self, in_key, out_key, model_filepath):
        self.in_key = in_key
        self.out_key = out_key
        f = open(model_filepath)
        model = cPickle.load(f)

        self.pool_strides = []
        self.pool_shapes = []
        self.kernel_strides = []
        self.kernel_shapes = []

        for layer in model.layers:
            if isinstance(layer, pylearn2.models.mlp.ConvRectifiedLinear):
                self.pool_strides.append(layer.pool_stride)
                self.pool_shapes.append(layer.pool_shape)
                self.kernel_shapes.append(layer.kernel_shape)
                self.kernel_strides.append(layer.kernel_stride)

        self.pool_strides.reverse()
        self.pool_shapes.reverse()
        self.kernel_strides.reverse()
        self.kernel_shapes.reverse()

    def init_dataset(self, dataset):

        heatmaps = self._run(dataset, 0)
        num_examples = dataset[self.in_key].shape[0]

        shape = (num_examples, heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2])

        dataset.create_dataset(self.out_key,
                               shape,
                               chunks=(10, heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]))

    def _run(self, dataset, index):

        def expand(heatmap, rescale_factor):
            shape = heatmap.shape
            new_shape = (shape[0] * rescale_factor[0], shape[1] * rescale_factor[1])
            return scipy.misc.imresize(heatmap, new_shape)


        heatmap = dataset[self.in_key][index]
        num_conv_layers = len(self.pool_shapes)
        num_channels = heatmap.shape[-1]

        # Intial scale factor.
        scale_factor = [1, 1]
        for layer_index in range(num_conv_layers):
            scale_factor[0] = scale_factor[0] * self.kernel_strides[layer_index][0] * self.pool_strides[layer_index][0]
            scale_factor[1] = scale_factor[1] * self.kernel_strides[layer_index][1] * self.pool_strides[layer_index][1]

        x_dim = heatmap.shape[0] * scale_factor[0]
        y_dim = heatmap.shape[1] * scale_factor[1]

        heatmap_out = np.zeros((x_dim, y_dim, num_channels))

        for channel_index in range(num_channels):
            heatmap_out[:,:,channel_index] = expand(heatmap[:,:,channel_index], scale_factor)

        """
                heatmap_out = np.zeros(heatmap.shape * scale_factor)
                #heatmap at this point is np.array of dim (20,20,14)
                for layer_index in range(num_conv_layers):
                    for channel_index in range(num_channels):
                        expanded_heatmap = np.zeros()
                        heatmap[:,:,channel_index] = expand(heatmap[:,:,channel_index], self.pool_strides[layer_index])
                        heatmap = expand(heatmap, self.kernel_strides[layer_index])

                #we want heatmap at this point to be (
        """
        return heatmap_out


class ConvolvePriors(ClassificationStage):

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

    def dataset_inited(self,dataset):
        return 'l_convolved_heatmaps' in dataset.keys()

    def init_dataset(self, dataset):

        l_gripper_conv, palm_conv, r_gripper_conv, l_gripper_out, palm_out, r_gripper_out = self._run(dataset, 0)

        shape = (900, l_gripper_conv.shape[1], l_gripper_conv.shape[2], l_gripper_conv.shape[3])
        chunk_size = (10, l_gripper_conv.shape[1], l_gripper_conv.shape[2], l_gripper_conv.shape[3])

        dataset.create_dataset("l_convolved_heatmaps", shape, chunks=chunk_size)
        dataset.create_dataset("r_convolved_heatmaps", shape, chunks=chunk_size)
        dataset.create_dataset("p_convolved_heatmaps", shape, chunks=chunk_size)

        shape = (900, palm_out.shape[0], palm_out.shape[1], 3)
        chunk_size = (10, palm_out.shape[0], palm_out.shape[1], 3)
        dataset.create_dataset("convolved_heatmaps", shape, chunks=chunk_size)

    def _run(self, dataset, index):
        heatmaps = dataset['rescaled_heatmaps'][index]
        # img_in_shape = (416, 576)
        # l_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 0], img_in_shape)
        # palm_obs = scipy.misc.imresize(heatmaps[:, :, 1], img_in_shape)
        # r_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 2], img_in_shape)

        l_gripper_obs = heatmaps[:, :, 0]
        palm_obs = heatmaps[:, :, 1]
        r_gripper_obs = heatmaps[:, :, 2]
        img_in_shape = heatmaps.shape[:-1]


        img_in = np.zeros((1, 1, img_in_shape[0], img_in_shape[1]), dtype=np.float32)

        img_in[:, :] = l_gripper_obs
        l_gripper_conv = self.f(img_in)

        img_in[:, :] = r_gripper_obs
        r_gripper_conv = self.f(img_in)

        img_in[:, :] = palm_obs
        palm_conv = self.f(img_in)

        out_shape = palm_conv[0, 0].shape
        x_border = (img_in_shape[0]-out_shape[0])/2
        y_border = (img_in_shape[1]-out_shape[1])/2

        l_gripper_obs2 = l_gripper_obs[x_border:-x_border - 1, y_border:-y_border - 1]
        palm_obs2 = palm_obs[x_border:-x_border - 1, y_border:-y_border - 1]
        r_gripper_obs2 = r_gripper_obs[x_border:-x_border - 1, y_border:-y_border - 1]

        l_gripper_out = l_gripper_obs2 * palm_conv[0, 0] * r_gripper_conv[0, 1]
        palm_out = palm_obs2 * l_gripper_conv[0, 2] * r_gripper_conv[0, 3]
        r_gripper_out = r_gripper_obs2 * l_gripper_conv[0, 4] * palm_conv[0, 5]

        return l_gripper_conv, palm_conv, r_gripper_conv, l_gripper_out, palm_out, r_gripper_out

    def run(self, dataset, index):

        l_gripper_conv, palm_conv, r_gripper_conv, l_gripper_out, palm_out, r_gripper_out = self._run(dataset, index)

        dataset["l_convolved_heatmaps"][index] = l_gripper_conv
        dataset["r_convolved_heatmaps"][index] = r_gripper_conv
        dataset["p_convolved_heatmaps"][index] = palm_conv

        dataset['convolved_heatmaps'][index, :, :, 0] = l_gripper_out
        dataset['convolved_heatmaps'][index, :, :, 1] = palm_out
        dataset['convolved_heatmaps'][index, :, :, 2] = r_gripper_out


# class CalculateMax(GraspClassificationStage):
#
#
#     def run(self, dataset, index):
#
#         out_shape = dataset['convolved_heatmaps'][index][:, :, 0].shape
#         l_gripper_out = dataset['convolved_heatmaps'][index][:, :, 0]
#         palm_out = dataset['convolved_heatmaps'][index][:, :, 1]
#         r_gripper_out = dataset['convolved_heatmaps'][index][:, :, 2]
#         rgb_with_grasp = dataset["best_grasp"][index]
#
#         img_in_shape = dataset["rgbd_data"][index, :, :, 0].shape
#         x_border = (img_in_shape[0]-out_shape[0])/2
#         y_border = (img_in_shape[1]-out_shape[1])/2
#
#         rgb_with_grasp[:] = np.copy(dataset["rgbd_data"][index, x_border:-x_border - 1, y_border:-y_border - 1, 0:3])
#
#         l_max = np.argmax(l_gripper_out)
#         p_max = np.argmax(palm_out)
#         r_max = np.argmax(r_gripper_out)
#
#         lim = out_shape[1]
#         l_max_x, l_max_y = (l_max / lim, l_max % lim)
#         p_max_x, p_max_y = (p_max / lim, p_max % lim)
#         r_max_x, r_max_y = (r_max / lim, r_max % lim)
#
#         try:
#             rgb_with_grasp[l_max_x-5:l_max_x + 5, l_max_y-5:l_max_y + 5] = [0, 0, 0]
#             rgb_with_grasp[p_max_x-5:p_max_x + 5, p_max_y-5:p_max_y + 5] = [0, 0, 0]
#             rgb_with_grasp[r_max_x-5:r_max_x + 5, r_max_y-5:r_max_y + 5] = [0, 0, 0]
#         except:
#             pass
#
#         dataset['best_grasp'][index] = rgb_with_grasp
#
#
#
#
#
#
# class CalculateTopFive(GraspClassificationStage):
#
#     def __init__(self, input_key='convolved_heatmaps',
#                  output_key='dependent_grasp_points',
#                  border_dim=15):
#         self.input_key = input_key
#         self.output_key = output_key
#         self.border_dim = border_dim
#
#     def get_local_minima(self, output):
#         output2 = np.copy(output)
#         e = np.zeros(output2.shape)
#         extrema = argrelextrema(output2, np.greater)
#         for i in range(len(extrema[0])):
#             e[extrema[0][i], extrema[1][i]] = output[extrema[0][i], extrema[1][i]]
#
#         return e
#
#     def get_local_minima_above_threshold(self, heatmap):
#         extrema = self.get_local_minima(heatmap)
#
#         extrema_average = extrema.sum()/(extrema != 0).sum()
#         #threshold is mean of extrema excluding zeros times a scaling factor
#         threshold = extrema_average - .05 * extrema_average
#
#         #set anything negative to 0
#         extrema = np.where(extrema <= threshold, extrema, 0)
#
#         return extrema
#
#     def get_scaled_extremas(self, rgbd_img, heatmaps, extremas):
#         #extrema imposed on input:
#         border_dim = (2*self.border_dim, 2*self.border_dim)
#         extremas_with_border_shape = [sum(x) for x in zip(extremas.shape, border_dim)]
#         extremas_with_border = np.zeros(extremas_with_border_shape)
#
#         extremas_with_border[self.border_dim:-self.border_dim, self.border_dim:-self.border_dim] = heatmaps[:, :]
#         scaled_extremas = scipy.misc.imresize(extremas_with_border, rgbd_img.shape[0:2], interp='nearest')
#
#         return scaled_extremas
#
#     def run(self, dataset, index):
#
#         heatmaps = dataset[self.input_key][index]
#         rgbd_img = dataset['rgbd_data'][index]
#
#         grasp_points_img = copy.deepcopy(rgbd_img[:, :, 0:3])
#
#         for heatmap_index in range(3):
#             heatmap = heatmaps[:, :, heatmap_index]
#
#
#             local_minima = self.get_local_minima_above_threshold(heatmap)
#             local_minima = self.get_local_minima(heatmap)
#             scaled_extremas = self.get_scaled_extremas(rgbd_img, heatmap, local_minima)
#
#             extrema_dict = dict((e, i)
#                                for i, e in np.ndenumerate(scaled_extremas)
#                                if e > 0.0)
#
#             sorted_extremas = sorted(extrema_dict, key=lambda key: key, reverse=True)
#
#             for j, extrema in enumerate(sorted_extremas[-20:]):
#
#                 max_extrema = extrema_dict[extrema]
#                 heat_val = (j * 254 / 5)
#
#                 #top
#                 for i in range(-5, 5):
#                     grasp_points_img[max_extrema[0]-5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#                     grasp_points_img[max_extrema[0]-4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#                 #bot
#                 for i in range(-5, 5):
#                     grasp_points_img[max_extrema[0]+4, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#                     grasp_points_img[max_extrema[0]+5, max_extrema[1] + i][:3] = [0.0, heat_val, 0.0]
#                 #left
#                 for i in range(-5, 5):
#                     grasp_points_img[max_extrema[0]+i, max_extrema[1]-5][:3] = [0.0, heat_val, 0.0]
#                     grasp_points_img[max_extrema[0]+i, max_extrema[1]-4][:3] = [0.0, heat_val, 0.0]
#                 #right
#                 for i in range(-5, 5):
#                     grasp_points_img[max_extrema[0]+i, max_extrema[1]+5][:3] = [0.0, heat_val, 0.0]
#                     grasp_points_img[max_extrema[0]+i, max_extrema[1]+4][:3] = [0.0, heat_val, 0.0]
#
#             dataset[self.output_key][index, heatmap_index] = grasp_points_img











