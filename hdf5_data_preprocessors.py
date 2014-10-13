
import h5py
import numpy as np
import math
import os

from pylearn2.datasets import preprocessing
from pylearn2.expr.preprocessing import global_contrast_normalize


class ExtractRawGraspData(preprocessing.Preprocessor):

    def __init__(self, raw_data_folder,  data_labels=("rgbd_patches", "patch_labels")):
        self.raw_data_folder = raw_data_folder
        self.data_labels = data_labels

    def apply(self, dataset, can_fit=False):

        #check if we have already extracted the raw data
        if self.data_labels[0] in dataset.keys() or self.data_labels[1] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        dataset.create_dataset(self.data_labels[0], (0, 480, 640, 4), chunks=(100, 480, 640, 4), maxshape=(None, 480, 640, 4))
        dataset.create_dataset(self.data_labels[1], (0, 480, 640), chunks=(100, 480, 640), maxshape=(None, 480, 640))

        current_total = 0
        for subdirectory in os.walk(self.raw_data_folder).next()[1]:

            print subdirectory

            old_dataset = h5py.File(self.raw_data_folder + subdirectory + '/rgbd_and_labels.h5')
            number_to_add = old_dataset['rgbd'].shape[0]

            for label in self.data_labels:
                new_size = list(dataset[label].shape)
                new_size[0] += number_to_add
                new_size = tuple(new_size)
                dataset[label].resize(new_size)
                dataset[label][current_total: current_total+number_to_add] = old_dataset[label][:]

            current_total += number_to_add


class SplitGraspPatches(preprocessing.Preprocessor):

    def __init__(self,
                 source_dataset_filepath,
                 output_keys=(("train_patches", "train_patch_labels"), ("valid_patches", "valid_patch_labels"), ("test_patches", "test_patch_labels")),
                 output_weights = (.8, .1, .1),
                 source_keys=("rgbd_patches", "rgbd_patch_labels")):

        self.source_dataset = h5py.File(source_dataset_filepath)
        self.output_keys = output_keys
        #normalize the output weights
        self.output_weights = [x/sum(output_weights) for x in output_weights]
        self.source_keys = source_keys

    def apply(self, dataset, can_fit=False):
        #check if we have already extracted patches for this set of patch_labels
        if self.output_keys[0][0] in dataset.keys():
            print "skipping split_patches, this has already been run"
            return

        for index in range(len(self.output_keys)):
            output_key_pair = self.output_keys[index]
            patch_key = output_key_pair[0]
            label_key = output_key_pair[1]

            num_patches = math.floor(self.output_weights[index] * self.source_dataset[self.source_keys[0]].shape[0])
            num_patches = num_patches - (num_patches % 20)
            patch_shape = self.source_dataset[self.source_keys[0]].shape[1:4]

            #dataset.create_dataset(patch_key, (num_patches, patch_shape[0], patch_shape[1], patch_shape[2]), chunks=(100, patch_shape[0], patch_shape[1], patch_shape[2]))
            #dataset.create_dataset(label_key, (num_patches, 1))

            start_range = math.floor(sum(self.output_weights[:index])*self.source_dataset[self.source_keys[0]].shape[0])

            dataset[patch_key] = self.source_dataset[self.source_keys[0]][int(start_range):int(start_range+num_patches)]
            dataset[label_key] = self.source_dataset[self.source_keys[1]][int(start_range):int(start_range+num_patches)]





class ExtractGraspPatches(preprocessing.Preprocessor):

    def __init__(self,
                 patch_shape=(25, 25),
                 patch_labels=("rgbd_patches", "patch_labels"),
                 patch_source_labels=("rgbd", "labels"),
                 num_patches=100000):

        self.patch_shape = patch_shape
        self.patch_labels = patch_labels
        self.patch_source_labels = patch_source_labels
        self.num_patches = num_patches

    def apply(self, dataset, can_fit=False):
        #check if we have already extracted patches for this set of patch_labels
        if self.patch_labels[0] in dataset.keys() or self.patch_labels[1] in dataset.keys():
            print "skipping extract_patches, this has already been run"
            return

        dataset.create_dataset(self.patch_labels[0], (self.num_patches, self.patch_shape[0], self.patch_shape[1], 4), chunks=(100, self.patch_shape[0], self.patch_shape[1], 4))
        dataset.create_dataset(self.patch_labels[1], (self.num_patches, 1), chunks=(100, 1))


        X = dataset[self.patch_source_labels[0]]
        y = dataset[self.patch_source_labels[1]]

        num_images = X.shape[0]

        patch_X = dataset[self.patch_labels[0]]
        patch_y = dataset[self.patch_labels[1]]

        channel_slice = slice(0, X.shape[-1])

        rng = preprocessing.make_np_rng([1, 2, 3], which_method="randint")

        i = 0
        iteration_count = 0
        while i < self.num_patches:
            if iteration_count % 1000 == 0:
                print "extracting patches: " + str(i) + "/" + str(self.num_patches)

            image_num = rng.randint(num_images)
            x_args = [image_num]
            y_args = [image_num]

            for x_index in range(y.shape[1]):
                for y_index in range(y.shape[2]):
                    if x_index < y.shape[1]-self.patch_shape[0] and y_index < y.shape[2]-self.patch_shape[1]:
                        if y[image_num, x_index, y_index] > 0:

                            x_args.append(slice(x_index, x_index + self.patch_shape[0]))
                            y_args.append(x_index + self.patch_shape[0]/2.0)

                            x_args.append(slice(y_index, y_index + self.patch_shape[1]))
                            y_args.append(y_index + self.patch_shape[1]/2.0)

                            x_args.append(channel_slice)

                            patch_X[i] = X[tuple(x_args)]
                            patch_y[i] = y[tuple(y_args)]

                            x_args = [image_num]
                            y_args = [image_num]
                            i += 1

                            if i == self.num_patches:
                                return
            iteration_count += 1



class PerChannelGlobalContrastNormalizePatches(preprocessing.Preprocessor):

    def __init__(self,
                 data_to_normalize_key,
                 normalized_data_key,
                 batch_size,
                 subtract_mean=True,
                 scale=1.,
                 sqrt_bias=0.,
                 use_std=False,
                 min_divisor=1e-8):

        self.data_to_normalize_key = data_to_normalize_key
        self.normalized_data_key = normalized_data_key
        self.batch_size = batch_size
        self.subtract_mean = subtract_mean
        self.scale = scale
        self.sqrt_bias = sqrt_bias
        self.use_std = use_std
        self.min_divisor = min_divisor

    def apply(self, dataset, can_fit=False):

        #check if we have already flattened patches
        if self.normalized_data_key in dataset.keys():
            print "skipping normalization, this has already been run"
            return
        else:
            print "normalizing patches"

        in_data = dataset[self.data_to_normalize_key]
        data_size = in_data.shape[0]

        dataset.create_dataset(self.normalized_data_key, in_data.shape, chunks=((self.batch_size,)+in_data.shape[1:]))

        out_data = dataset[self.normalized_data_key]

        #iterate over patches
        for patch_index in range(data_size):
            if patch_index % 2000 == 0:
                print str(patch_index) + '/' + str(data_size)

            #iterate over rgbd so they are all normalized separately at this point
            for channel in range(4):
                out_data[patch_index, :, :, channel] = global_contrast_normalize(in_data[patch_index, :, :, channel],
                                                                             scale=self.scale,
                                                                             subtract_mean=self.subtract_mean,
                                                                             use_std=self.use_std,
                                                                             sqrt_bias=self.sqrt_bias,
                                                                             min_divisor=self.min_divisor)


class MakeC01B(preprocessing.Preprocessor):

    def __init__(self, data_labels=("train_patches", "test_patches", "valid_patches"), y_labels=("train_patch_labels", "test_patch_labels", "valid_patch_labels")):

        self.data_labels = data_labels
        self.y_labels = y_labels

    def apply(self, dataset, can_fit=False):

        #check if we have already extracted the raw data
        if "c01b_" + self.data_labels[0] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        for index in range(len(self.data_labels)):

            data_label = self.data_labels[index]
            y_label = self.y_labels[index]

            print data_label
            num_images = dataset[data_label].shape[0]
            x_dim = dataset[data_label].shape[1]
            y_dim = dataset[data_label].shape[2]
            dataset.create_dataset("c01b_" + data_label, (4, x_dim, y_dim, num_images), chunks=(4, x_dim, y_dim, 4))
            dataset["c01b_" + y_label] = dataset[y_label]

            for i in range(num_images):
                if i % (num_images/100.0) == 0:
                    print "converting to co1b: " + str(i) + "/" + str(num_images)

                    b01c_data = np.copy(dataset[data_label][i])
                    c01b_data = np.rollaxis(b01c_data, 2)

                    dataset["c01b_" + data_label][:, :, :, i] = c01b_data
