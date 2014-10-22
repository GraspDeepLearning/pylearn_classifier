
import h5py
import os

from pylearn2.datasets import preprocessing

import hdf5_data_preprocessors
import paths


def preprocess_grasp_dataset(attribs):

    pipeline = preprocessing.Pipeline()

    pipeline.items.append(hdf5_data_preprocessors.CopyInRaw(
        source_dataset_filepath=attribs["raw_filepath"],
        input_keys=('patches', 'labels'),
        output_keys=('patches', 'patch_labels')

    ))

    pipeline.items.append(hdf5_data_preprocessors.RandomizePatches(
        keys=('patches', 'patch_labels')
    ))

    pipeline.items.append(hdf5_data_preprocessors.LecunSubtractiveDivisiveLCN(in_key='patches', out_key='normalized_patches'))

    #now we split the patches up into train, test, and valid sets
    pipeline.items.append(hdf5_data_preprocessors.SplitGraspPatches(
        output_keys=(("train_patches", "train_patch_labels"), ("valid_patches", "valid_patch_labels"), ("test_patches", "test_patch_labels")),
        output_weights=(.8, .1, .1),

        source_keys=("normalized_patches", "patch_labels")))

    #now we swap around the axis so the data fits nicely onto the gpu
    # C01B rather than B01C
    pipeline.items.append(hdf5_data_preprocessors.MakeC01B())

    #now lets actually make a new dataset and run it through the pipeline
    if not os.path.exists(paths.PYLEARN_DATA_PATH + "deep_learning_grasp_data"):
        os.makedirs(paths.PYLEARN_DATA_PATH + "deep_learning_grasp_data")

    hd5f_dataset = h5py.File(attribs["output_filepath"])
    pipeline.apply(hd5f_dataset)


if __name__ == "__main__":

    raw_rgbd_datafile = paths.choose_from(paths.RAW_TRAINING_DATASET_DIR)
    raw_rgbd_filepath = paths.RAW_TRAINING_DATASET_DIR + raw_rgbd_datafile

    preprocess_attribs = dict(sets=("train", "test", "valid"),
                              patch_shape=(170, 170),
                              raw_filepath=raw_rgbd_filepath,
                              output_filepath=paths.PROCESSED_TRAINING_DATASET_DIR + 'processed_per_channel_0_to_1_normalization3' + raw_rgbd_datafile)

    preprocess_grasp_dataset(preprocess_attribs)
