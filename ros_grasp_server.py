#!/usr/bin/env python
import rospy
from pylearn_classifier_gdl.srv import CalculateGraspsService
import choose
import paths
from classification_pipelines import *

import cPickle
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

import numpy as np
import os
import h5py
import time


def init_save_file(input_data_file, input_model_file):

    dataset_filepath = paths.HEATMAPS_DATASET_DIR + 'temp/' + input_data_file + '_' + input_model_file + '.h5'

    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)

    save_dset = h5py.File(dataset_filepath)

    return save_dset, dataset_filepath


def init_rgbd_file(dataset_file):

    raw_rgbd_filepath = paths.RAW_TRAINING_DATASET_DIR + "temp/" + dataset_file + '.h5'

    if os.path.exists(raw_rgbd_filepath):
        os.remove(raw_rgbd_filepath)

    input_dset = h5py.File(raw_rgbd_filepath)

    return input_dset, raw_rgbd_filepath


class GraspServer:

    def __init__(self):
        self.pylearn_model = None

        conv_model_name = "processed_just_tobasco_with_wrist_roll_5_layer_170x170_12_11_11_55"
        conv_model_filepath = paths.MODEL_DIR + conv_model_name + "/cnn_model.pkl"

        dataset_file = str(int(round(time.time() * 1000)))

        input_dset, raw_rgbd_filepath = init_rgbd_file(dataset_file)
        self.input_dset = input_dset
        self.input_dset.create_dataset("rgbd", (1, 480, 640, 4))

        save_dset, save_filepath = init_save_file(dataset_file, conv_model_name)
        self.save_dset = save_dset
        self.save_dset.create_dataset("mask", (480, 640))

        rospy.logout(raw_rgbd_filepath)
        rospy.logout(save_filepath)
        self.pipeline = BarrettGraspClassificationPipeline(save_filepath, raw_rgbd_filepath, conv_model_filepath, input_key="rgbd")
        self.pipeline.run()

        self.service = rospy.Service('calculate_grasps_service', CalculateGraspsService, self.service_request_handler)

        rospy.spin()

    def service_request_handler(self, request):
        rospy.loginfo("received request")
        rgbd = np.array(request.rgbd).reshape((480, 640, 4))
        mask = np.array(request.mask).reshape((480, 640))

        self.input_dset["rgbd"][0] = rgbd
        self.save_dset["mask"] = mask

        self.pipeline.run()

        out = self.save_dset['independent_x_priors']



        return []
    


if __name__ == "__main__":

    rospy.init_node('grasp_server_node')
    rospy.loginfo("starting grasp_server_node!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    grasp_server = GraspServer()
