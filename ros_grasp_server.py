#!/usr/bin/env python
import rospy

import choose
import paths
from classification_pipelines import *

import cPickle
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from uvd_xyz_conversion.srv import UVDTOXYZ
from pylearn_classifier_gdl.srv import CalculateGraspsService, CalculateGraspsServiceResponse
from gdl_grasp_msgs.msg import Grasp

import tf

import numpy as np
import os
import h5py
import time
import pickle

from sensor_msgs.msg import JointState

from grasp_priors import GraspPriorsList


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

        try:
            rospy.wait_for_service('/uvd_to_xyz', 6)
        except rospy.ServiceException as exc:
            rospy.logerr("uvd_to_xyz service never came up")
            assert False

        self.uvd_to_xyz_proxy = rospy.ServiceProxy('uvd_to_xyz', UVDTOXYZ)

        f = open("grasp_priors_list.pkl")
        self.grasp_priors_list = pickle.load(f)

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
        self.save_dset["mask"][:, :] = mask

        self.pipeline.run()

        grasps = self.pipeline._pipeline_stages[-1].grasps

        grasp_msgs = CalculateGraspsServiceResponse()

        for grasp in grasps:
            grasp_energy, grasp_type,  palm_index, argmax_u, argmax_v = grasp
            d = rgbd[argmax_u, argmax_v, 3]

            resp = self.uvd_to_xyz_proxy(argmax_u, argmax_v, d)
            x = resp.x
            y = resp.y
            z = resp.z

            joint_values = self.grasp_priors_list.get_grasp_prior(grasp_type).joint_values
            wrist_roll = self.grasp_priors_list.get_grasp_prior(grasp_type).wrist_roll

            quat = tf.transformations.quaternion_from_euler(0, 0, wrist_roll)

            grasp_msg = Grasp()
            grasp_msg.pose.position.x = x
            grasp_msg.pose.position.y = y
            grasp_msg.pose.position.z = z
            grasp_msg.pose.orientation.x = quat[0]
            grasp_msg.pose.orientation.y = quat[1]
            grasp_msg.pose.orientation.z = quat[2]
            grasp_msg.pose.orientation.w = quat[3]


            jv = JointState()
            jv.name = ["bhand/finger_1/prox_joint",
                       "bhand/finger_1/med_joint",
                       "bhand/finger_1/dist_joint",
                       "bhand/finger_2/prox_joint"
                       "bhand/finger_2/med_joint",
                       "bhand/finger_2/dist_joint",
                       "bhand/finger_3/med_joint",
                       "bhand/finger_3/dist_joint"]
            jv.position = joint_values
            grasp_msg.joint_values = jv
            grasp_msg.grasp_energy = grasp_energy

            grasp_msg.grasp_type = grasp_type

            grasp_msgs.grasps.append(grasp_msg)

        rospy.loginfo("finished")

        return grasp_msgs

if __name__ == "__main__":

    rospy.init_node('grasp_server_node')
    rospy.loginfo("starting grasp_server_node...")
    grasp_server = GraspServer()
