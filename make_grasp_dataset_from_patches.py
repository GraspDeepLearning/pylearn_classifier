
import h5py
import os

from pylearn2.datasets import preprocessing

import hdf5_data_preprocessors

import paths


#the dataset we are going to train the model against
def get_raw_rgbd():

    raw_rgbd_data_files = os.listdir(paths.RAW_TRAINING_DATASET_DIR)

    print
    print "Choose raw input: "
    print

    for i in range(len(raw_rgbd_data_files)):
        print str(i) + ": " + raw_rgbd_data_files[i]

    print
    raw_rgbd_index = int(raw_input("Enter Id of rgbd_images to preprocess (ex 0, 1, or 2): "))

    raw_rgbd_data_file = raw_rgbd_data_files[raw_rgbd_index]

    return raw_rgbd_data_file


def preprocess_grasp_dataset(attribs):

    pipeline = preprocessing.Pipeline()

    pipeline.items.append(hdf5_data_preprocessors.SplitGraspPatches(
        source_dataset_filepath=attribs["raw_filepath"],
        output_keys=(("train_patches", "train_patch_labels"), ("valid_patches", "valid_patch_labels"), ("test_patches", "test_patch_labels")),
        output_weights=(.8, .1, .1),
        source_keys=("rgbd_patches", "rgbd_patch_labels")))

    pipeline.items.append(hdf5_data_preprocessors.MakeC01B())

    #now lets actually make a new dataset and run it through the pipeline
    if not os.path.exists(paths.PYLEARN_DATA_PATH + "deep_learning_grasp_data"):
        os.makedirs(paths.PYLEARN_DATA_PATH + "deep_learning_grasp_data")

    hd5f_dataset = h5py.File(attribs["output_filepath"])
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    raw_rgbd_datafile = get_raw_rgbd()

    preprocess_attribs = dict(sets=("train", "test", "valid"),
                              patch_shape=(72, 72),
                              raw_filepath= paths.RAW_TRAINING_DATASET_DIR + raw_rgbd_datafile,
                              output_filepath=paths.PROCESSED_TRAINING_DATASET_DIR + 'processed_' + raw_rgbd_datafile)

    preprocess_grasp_dataset(preprocess_attribs)
