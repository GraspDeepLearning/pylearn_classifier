
import os

#this is the root directory of the data folder
PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]

#this points to the location off all the raw rgbd images
#these were either from:
#1)gazebo i.e. graspit data
#2)download i.e saxena data
RAW_TRAINING_DATASET_DIR = PYLEARN_DATA_PATH + '/raw_rgbd_images/'

#this is where datasets that have been processed go.
#these datasets have have patches extracted, data normalized, etc
PROCESSED_TRAINING_DATASET_DIR = PYLEARN_DATA_PATH + 'deep_learning_grasp_data/'

#this points to the locations of all the trained models
MODEL_DIR = '~/grasp_deep_learning/pylearn_classifier_gdl/models/'

#this is the directory where we get the model yaml file and hyper parameters from.
MODEL_TEMPLATE_DIR = "~/grasp_deep_learning/pylearn_classifier_gdl/model_templates/"

#this is the output from running a model on a dataset
HEATMAPS_DATASET_DIR = PYLEARN_DATA_PATH + '/heatmaps/'


