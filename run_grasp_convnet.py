
from grasp_classification_pipeline import *

import paths
import os

RUNNING_GRASPS = True


def init_save_file(input_data_file, input_model_file):

    dataset_filepath = paths.HEATMAPS_DATASET_DIR + input_data_file[:-3] + '_' + input_model_file + '4.h5'

    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)

    h5py.File(dataset_filepath)

    return dataset_filepath


def main():

    conv_model_name = paths.choose_from(paths.MODEL_DIR)
    conv_model_filepath = paths.MODEL_DIR + conv_model_name + "/cnn_model.pkl"

    dataset_file = paths.choose_from(paths.RAW_TRAINING_DATASET_DIR)
    raw_rgbd_filepath = paths.RAW_TRAINING_DATASET_DIR + dataset_file


    save_filepath = init_save_file(dataset_file, conv_model_name)

    pipeline = GraspClassificationPipeline(save_filepath, raw_rgbd_filepath)



    pipeline.run()

if __name__ == "__main__":
    main()
