
import h5py
import numpy as np
import os
from scipy import misc

from pylearn2.datasets import preprocessing
from pylearn2.expr.preprocessing import global_contrast_normalize


#these borrow heavily from the pylearn2 preprocessing classes, just modified slightly
#to work with hdf5 and my data


class ExtractRawNYUData(preprocessing.Preprocessor):

    def __init__(self, raw_data_filepath,  data_labels=("rgbd", "labels")):
        self.raw_data_filepath = raw_data_filepath
        self.data_labels = data_labels

    def apply(self, dataset, can_fit=False):

        raw_dataset = h5py.File(self.raw_data_filepath)

        #check if we have already extracted the raw data
        if self.data_labels[0] in dataset.keys() or self.data_labels[1] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        num_images = raw_dataset["images"].shape[0]

        dataset.create_dataset(self.data_labels[0], (num_images, 640, 480, 4), chunks=(10, 640, 480, 4))
        dataset.create_dataset(self.data_labels[1], (num_images, 640, 480), chunks=(10, 640, 480))

        for i in range(num_images):
            if i % (num_images/10) == 0:
                print "extracting raw data: " + str(i) + "/" + str(num_images)

            dataset[self.data_labels[0]][i, :, :, 0:3] = np.rollaxis(raw_dataset["images"][i], 0, 3)
            dataset[self.data_labels[0]][i, :, :, 3] = raw_dataset["depths"][i]
            dataset[self.data_labels[1]][i] = raw_dataset["labels"][i]


class ExtractRawUWashData(preprocessing.Preprocessor):

    def __init__(self, raw_data_folder,  data_labels=("rgbd_patches", "patch_labels")):
        self.raw_data_folder = raw_data_folder
        self.data_labels = data_labels

    def apply(self, dataset, can_fit=False):

        #check if we have already extracted the raw data
        if self.data_labels[0] in dataset.keys() or self.data_labels[1] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        print "determining number of labels"
        num_images = 1
        label_list = []

        for root, subfolders, files in os.walk(self.raw_data_folder):
            for data_file in files:
                if data_file.endswith("_crop.png"):
                    num_images += 1
                    label_list.append(root.split("/")[-1])

        unique_label_list = list(set(label_list))

        print "num_images: " + str(len(label_list))
        print "num_labels: " + str(len(unique_label_list))

        print "creating datasets"
        dataset.create_dataset("label_id_to_string", (len(unique_label_list), 1), dtype=h5py.special_dtype(vlen=unicode))
        dataset.create_dataset(self.data_labels[0], (num_images, 72, 72, 4), maxshape=(None, 72, 72, 4), chunks=(100, 72, 72, 4))
        dataset.create_dataset(self.data_labels[1], (num_images, 1), maxshape=(None, 1), chunks=(100, 1))

        for i in range(len(unique_label_list)):
            dataset["label_id_to_string"][i] = unique_label_list[i]

        print "converting data to h5py format"
        total_count = 0
        for root, subfolders, files in os.walk(self.raw_data_folder):

            #print the current directory we are working on
            #as well as how far along we are
            print root
            print str(total_count) + "/" + str(num_images)

            rgb_file_list = []
            depth_file_list = []
            label_list = []

            for data_file in files:
                if data_file.endswith("_crop.png"):
                    rgb_file_list.append(root + "/" + data_file)
                    depth_file_list.append(root + '/' + data_file[:-8] + "depthcrop.png")
                    label_list.append(root.split("/")[-1])

            for i in range(len(rgb_file_list)):
                rgb_im = misc.imread(rgb_file_list[i])
                depth_im = misc.imread(depth_file_list[i])

                #we are going to scale the images so
                #smallest dimension is 72
                if rgb_im.shape[0] > rgb_im.shape[1]:
                    scale_factor = 72.0/rgb_im.shape[1]
                else:
                    scale_factor = 72.0/rgb_im.shape[0]

                rgb_im_resized = misc.imresize(rgb_im, scale_factor)
                depth_im_resized = misc.imresize(depth_im, scale_factor)

                #now we are going to crop out a 72x72 patch from the center of
                #the resized image
                center_x, center_y = rgb_im_resized.shape[0]/2, rgb_im_resized.shape[1]/2

                rgb_im_cropped = rgb_im_resized[center_x-36:center_x+36, center_y-36:center_y+36, :]
                depth_im_cropped = depth_im_resized[center_x-36:center_x+36, center_y-36:center_y+36]

                dataset[self.data_labels[0]][i+total_count, :, :, 0:3] = rgb_im_cropped
                dataset[self.data_labels[0]][i+total_count, :, :, 3] = depth_im_cropped
                dataset[self.data_labels[1]][i+total_count] = unique_label_list.index(label_list[i])

            total_count += len(rgb_file_list)

class ExtractRawGraspData(preprocessing.Preprocessor):

    def __init__(self, raw_data_folder,  data_labels=("rgbd_patches", "patch_labels")):
        self.raw_data_folder = raw_data_folder
        self.data_labels = data_labels

    def apply(self, dataset, can_fit=False):

        #check if we have already extracted the raw data
        if self.data_labels[0] in dataset.keys() or self.data_labels[1] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        print "determining number of labels"
        num_images = 0
        label_list = []

        for root, subfolders, files in os.walk(self.raw_data_folder):
            for data_file in files:
                if data_file.endswith("overlay.png"):
                    num_images += 1


        print "creating datasets"
        dataset.create_dataset(self.data_labels[0], (num_images, 480, 640, 4), chunks=(10, 480, 640, 4))
        dataset.create_dataset(self.data_labels[1], (num_images, 480, 640), chunks=(10, 480, 640))


        print "converting data to h5py format"
        total_count = 0
        for root, subfolders, files in os.walk(self.raw_data_folder):

            #print the current directory we are working on
            #as well as how far along we are
            print root
            print str(total_count) + "/" + str(num_images)

            r_file_list = []
            g_file_list = []
            b_file_list = []
            d_file_list = []

            label_list = []

            for data_file in files:
                if data_file.endswith("overlay.png"):
                    r_file_list.append(root + "/r.png")
                    g_file_list.append(root + "/g.png")
                    b_file_list.append(root + "/b.png")
                    d_file_list.append(root + "/d.png")
                    label_list.append(root + "/out.png")

            for i in range(len(r_file_list)):
                r_im = misc.imread(r_file_list[i])
                g_im = misc.imread(g_file_list[i])
                b_im = misc.imread(b_file_list[i])
                d_im = misc.imread(d_file_list[i])

                out_im = misc.imread(label_list[i])

                dataset[self.data_labels[0]][i+total_count, :, :, 0] = r_im
                dataset[self.data_labels[0]][i+total_count, :, :, 1] = g_im
                dataset[self.data_labels[0]][i+total_count, :, :, 2] = b_im
                dataset[self.data_labels[0]][i+total_count, :, :, 3] = d_im

                dataset[self.data_labels[1]][i+total_count, :, :] = out_im

            total_count += len(r_file_list)


class ExtractRawGraspData2(preprocessing.Preprocessor):

    def __init__(self, raw_data_folder,  data_labels=("rgbd_patches", "patch_labels")):
        self.raw_data_folder = raw_data_folder
        self.data_labels = data_labels

    def apply(self, dataset, can_fit=False):

        #check if we have already extracted the raw data
        if self.data_labels[0] in dataset.keys() or self.data_labels[1] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        old_dataset = h5py.File(self.raw_data_folder)

        for label in self.data_labels:
            dataset[label] = old_dataset[label][:]


class ExtractPatches(preprocessing.Preprocessor):

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

        #just rows and columns
        num_topological_dimensions = 2
        dataset.create_dataset(self.patch_labels[0], (self.num_patches, self.patch_shape[0], self.patch_shape[1], 4), chunks=(100, self.patch_shape[0], self.patch_shape[1], 4))
        dataset.create_dataset(self.patch_labels[1], (self.num_patches, 1))

        X = dataset[self.patch_source_labels[0]]
        y = dataset[self.patch_source_labels[1]]

        num_images = X.shape[0]

        patch_X = dataset[self.patch_labels[0]]
        patch_y = dataset[self.patch_labels[1]]

        channel_slice = slice(0, X.shape[-1])

        rng = preprocessing.make_np_rng([1, 2, 3], which_method="randint")

        for i in xrange(self.num_patches):
            if i % (self.num_patches/10) == 0:
                print "extracting patches: " + str(i) + "/" + str(self.num_patches)

            image_num = rng.randint(num_images)
            x_args = [image_num]
            y_args = [image_num]

            for j in xrange(num_topological_dimensions):
                max_coord = X.shape[j + 1] - self.patch_shape[j]
                coord = rng.randint(max_coord + 1)
                x_args.append(slice(coord, coord + self.patch_shape[j]))
                y_args.append(coord + self.patch_shape[j]/2.0)

            x_args.append(channel_slice)

            patch_X[i] = X[tuple(x_args)]
            patch_y[i] = y[tuple(y_args)]



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

        #just rows and columns
        num_topological_dimensions = 2
        dataset.create_dataset(self.patch_labels[0], (self.num_patches, self.patch_shape[0], self.patch_shape[1], 4), chunks=(100, self.patch_shape[0], self.patch_shape[1], 4))
        dataset.create_dataset(self.patch_labels[1], (self.num_patches, 1))

        X = dataset[self.patch_source_labels[0]]
        y = dataset[self.patch_source_labels[1]]

        num_images = X.shape[0]

        patch_X = dataset[self.patch_labels[0]]
        patch_y = dataset[self.patch_labels[1]]

        channel_slice = slice(0, X.shape[-1])

        rng = preprocessing.make_np_rng([1, 2, 3], which_method="randint")

        i = 0
        while i < self.num_patches:
            if i % (self.num_patches/10) == 0:
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

                            print i

                            x_args = [image_num]
                            y_args = [image_num]
                            i += 1

                            if i == self.num_patches:
                                return




class FlattenPatches(preprocessing.Preprocessor):

    def __init__(self,
                 patch_label="rgbd_patches",
                 flattened_patch_label="rgbd_flattened_patches"):
        self.patch_label = patch_label
        self.flattened_patch_label = flattened_patch_label

    def apply(self, dataset, can_fit=False):

        #check if we have already flattened patches
        if self.flattened_patch_label in dataset.keys():
            print "skipping flatten_patches, this has already been run"
            return
        else:
            print "flattening_patches"

        patches = dataset[self.patch_label]
        num_patches = patches.shape[0]
        num_flattened_features = 1
        for i in range(1, len(patches.shape)):
            num_flattened_features *= patches.shape[i]

        dataset.create_dataset(self.flattened_patch_label, (num_patches, num_flattened_features), chunks=(100, num_flattened_features))

        flattened_patches = dataset[self.flattened_patch_label]
        for i in range(num_patches):
            flattened_patches[i] = patches[i].flatten()


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


class SplitData(preprocessing.Preprocessor):

    def __init__(self,
                 data_to_split_key=('rgbd_patches', 'patch_labels'),
                 sets=("test", "valid"),
                 patch_shape=(72, 72),
                 num_patches_per_set=(10000, 10000)):

        self.data_to_split_key = data_to_split_key
        self.sets = sets
        self.patch_shape = patch_shape
        self.num_patches_per_set = num_patches_per_set

    def apply(self, dataset, can_fit=False):

        for set_index in range(len(self.sets)):

            set_type = self.sets[set_index]

            if set_type + "_patches" in dataset.keys():
                print "skipping " + set_type + "already exists in dataset"
                continue
            else:
                print "extracting " + set_type

            set_shape = (self.num_patches_per_set[set_index],) + self.patch_shape + (4,)
            dataset.create_dataset(set_type + "_patches", set_shape, chunks=((100,) + set_shape[1:]))
            dataset.create_dataset(set_type + "_patch_labels", (self.num_patches_per_set[set_index], 1))

            num_original_patches = dataset[self.data_to_split_key[0]].shape[0]

            for i in range(set_shape[0]):
                if i % 2000 == 0:
                    print str(i) + "/" + str(set_shape[0])
                index = np.random.randint(num_original_patches)
                dataset[set_type + "_patches"][i] = dataset[self.data_to_split_key[0]][index]
                dataset[set_type + "_patch_labels"][i] = dataset[self.data_to_split_key[1]][index]


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

            dataset.create_dataset("c01b_" + data_label, (4, 72, 72, num_images), chunks=(4, 72, 72, 100))
            dataset["c01b_" + y_label] = dataset[y_label]

            for i in range(num_images):
                if i % (num_images/100) == 0:
                    print "converting to co1b: " + str(i) + "/" + str(num_images)

                    b01c_data = np.copy(dataset[data_label][i])
                    c01b_data = np.rollaxis(b01c_data, 2)

                    dataset["c01b_" + data_label][:, :, :, i] = c01b_data
