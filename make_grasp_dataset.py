
import h5py
import os

from pylearn2.datasets import preprocessing

import hdf5_data_preprocessors

PYLEARN_DATA_PATH = os.path.expanduser(os.environ["PYLEARN2_DATA_PATH"])


def preprocess_grasp_dataset(attribs):

    pipeline = preprocessing.Pipeline()

    # rgbd : (1449, 640, 480, 4)
    # labels: (1449, 640, 480)
    pipeline.items.append(hdf5_data_preprocessors.ExtractRawGraspData(attribs["raw_filepath"],
                                                                      data_labels=("rgbd", "labels")))

    #add the steps necessary to generate data for
    # valid, test and training datasets
    for i in range(len(attribs["sets"])):

        which_set = attribs["sets"][i]
        num_patches = attribs["num_patches_per_set"][i]

        #labels for the hdf5 file
        patch_label = which_set + "_patches"
        patch_labels = (patch_label, which_set + "_patch_labels")

        pipeline.items.append(hdf5_data_preprocessors.ExtractGraspPatches(patch_shape=attribs["patch_shape"],
                                                                          patch_labels=patch_labels,
                                                                          patch_source_labels=("rgbd", "labels"),
                                                                          num_patches=num_patches))

    pipeline.items.append(hdf5_data_preprocessors.MakeC01B())

    #now lets actually make a new dataset and run it through the pipeline
    if not os.path.exists(PYLEARN_DATA_PATH + "deep_learning_grasp_data"):
        os.makedirs(PYLEARN_DATA_PATH + "deep_learning_grasp_data")

    hd5f_dataset = h5py.File(attribs["output_filepath"])
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    preprocess_attribs = dict(sets=("train", "test", "valid"),
                              num_patches_per_set=(100000, 10000, 10000),
                              patch_shape=(72, 72),
                              raw_filepath=PYLEARN_DATA_PATH + "rgbd_images/",
                              output_filepath=PYLEARN_DATA_PATH + "deep_learning_grasp_data/rgbd_preprocessed_72x72.h5")

    preprocess_grasp_dataset(preprocess_attribs)
