
import h5py
import os

from pylearn2.datasets import preprocessing

import hdf5_data_preprocessors

PYLEARN_DATA_PATH = os.path.expanduser(os.environ["PYLEARN2_DATA_PATH"])


def preprocess_grasp_dataset(attribs):

    pipeline = preprocessing.Pipeline()

    pipeline.items.append(hdf5_data_preprocessors.SplitGraspPatches(
        source_dataset_filepath= attribs["raw_filepath"],
        output_keys=(("train_patches", "train_patch_labels"), ("valid_patches", "valid_patch_labels"), ("test_patches", "test_patch_labels")),
        output_weights= (.8, .1, .1),
        source_keys=("rgbd_patches", "rgbd_patch_labels")))

    pipeline.items.append(hdf5_data_preprocessors.MakeC01B())

    #now lets actually make a new dataset and run it through the pipeline
    if not os.path.exists(PYLEARN_DATA_PATH + "grasp_data"):
        os.makedirs(PYLEARN_DATA_PATH + "grasp_data")

    hd5f_dataset = h5py.File(attribs["output_filepath"])
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    preprocess_attribs = dict(sets=("train", "test", "valid"),
                              num_patches_per_set=(100000, 10000, 10000),
                              patch_shape=(72, 72),
                              raw_filepath=PYLEARN_DATA_PATH + "rgbd_images/coke_can/rgbd_and_labels.h5",
                              output_filepath=PYLEARN_DATA_PATH + "grasp_data/rgbd_preprocessed_72x72.h5")

    preprocess_grasp_dataset(preprocess_attribs)
