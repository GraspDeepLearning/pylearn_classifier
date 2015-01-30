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
import math

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

        conv_model_name = "processed_2.0m_7vc_barrett_18_grasp_types_5_layer_170x170_1_12_12_41"

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

    def service_request_handler(self, request):
        rospy.loginfo("received request")

        rgbd = np.array(request.rgbd).reshape((480, 640, 4))
        mask = np.array(request.mask).reshape((480, 640))

        self.input_dset["rgbd"][0] = rgbd
        self.save_dset["mask"][:, :] = mask

        self.pipeline._pipeline_stages[-1].mask = mask

        self.pipeline.run()

        grasps = self.pipeline._pipeline_stages[-1].grasps

        grasp_msgs = CalculateGraspsServiceResponse()

        for grasp in grasps:
            grasp_energy, grasp_type,  palm_index, argmax_v, argmax_u = grasp
            d = rgbd[argmax_v, argmax_u, 3]

            rospy.loginfo("u: " + str(argmax_u))
            rospy.loginfo("v: " + str(argmax_v))
            rospy.loginfo("d: " + str(d))

            resp = self.uvd_to_xyz_proxy(argmax_u, argmax_v, d)
            x = resp.x
            y = resp.y
            z = resp.z

            joint_values = self.grasp_priors_list.get_grasp_prior(grasp_type).joint_values
            wrist_roll = self.grasp_priors_list.get_grasp_prior(grasp_type).wrist_roll

            quat = tf.transformations.quaternion_from_euler(0, math.pi/2.0, wrist_roll, axes='szyx')
            #quat = tf.transformations.quaternion_from_euler(0, 0, 0, axes='szyx')

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


from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
import image_geometry
from sensor_msgs.msg import CameraInfo
from scipy.optimize import minimize

class RefineGrasps():

    def __init__(self, grasp_msg, palm_uvd, heatmaps, camera_param_topic='/camera/rgb/camera_info'):
        self.palm_uvd = palm_uvd
        self.grasp_msg = grasp_msg
        self.heatmaps = heatmaps

        self.joint_names = ["bhand/finger_1/prox_joint",
                            "bhand/finger_1/med_joint",
                            "bhand/finger_1/dist_joint",
                            "bhand/finger_2/prox_joint",
                            "bhand/finger_2/med_joint",
                            "bhand/finger_2/dist_joint",
                            "bhand/finger_3/med_joint",
                            "bhand/finger_3/dist_joint"]

        self.max_joint_limits = [3.14, 2.44, 0.84, 3.14, 2.44, 0.84, 2.44, 0.84]
        self.min_joint_limits = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.robot = URDF.load_from_parameter_server(verbose=False)
        self.tree = kdl_tree_from_urdf_model(self.robot)

        camera_info = rospy.wait_for_message(camera_param_topic, CameraInfo)
        self.pinhole_model = image_geometry.PinholeCameraModel()
        self.pinhole_model.fromCameraInfo(camera_info)

        base_link = 'bhand_palm_link'
        self.f_1_kdl_kin = KDLKinematics(self.robot, base_link, 'bhand_finger_1/dist_link')
        self.f_2_kdl_kin = KDLKinematics(self.robot, base_link, 'bhand_finger_2/dist_link')
        self.f_3_kdl_kin = KDLKinematics(self.robot, base_link, 'bhand_finger_3/dist_link')

    def xyz_to_uv(self, position):
        return self.pinhole_model.project3dToPixel(position)

    def cost_function(self, joint_values):

        f1_joint_angles = joint_values[0:3]
        f2_joint_angles = joint_values[3:6]
        f3_joint_angles = joint_values[6:8]

        #q = kdl_kin.random_joint_angles()
        f_1_pose = self.f_1_kdl_kin.forward(f1_joint_angles)
        f_2_pose = self.f_2_kdl_kin.forward(f2_joint_angles)
        f_3_pose = self.f_3_kdl_kin.forward(f3_joint_angles)

        f1_u, f1_v = self.xyz_to_uv(f_1_pose.position)
        f2_u, f2_v = self.xyz_to_uv(f_2_pose.position)
        f3_u, f3_v = self.xyz_to_uv(f_3_pose.position)
        p_u, p_v, _ = self.palm_uvd

        palm_energy = self.heatmaps[0, p_u, p_v]
        f_1_energy = self.heatmaps[0, f1_u, f1_v]
        f_2_energy = self.heatmaps[0, f2_u, f2_v]
        f_3_energy = self.heatmaps[0, f3_u, f3_v]

        return palm_energy + f_1_energy + f_2_energy + f_3_energy

    def run(self):
        # add a constraint for each joint in order to ensure they stay within
        # feasible joint range.
        constraints = []

        for i in range(len(self.grasp_msg.joint_values)):

            #joint values must be below max joint limit
            constraints.append({'type': 'ineq', 'fun': lambda x: np.array([x[i] - self.max_joint_limits[i]])})

            #joint values must be above min joint limit
            constraints.append({'type': 'ineq', 'fun': lambda x: np.array([x[i] - self.min_joint_limits[i]])})

        # the proximal joint for finger 1 and finger 2 must always be equal
        constraints.append({'type': 'ineq', 'fun': lambda x: np.array([x[0] - x[3]])})

        result = minimize(self.cost_function, self.grasp_msg.joint_values, constraints=constraints)

        final_joint_values = result.x
        final_energy = result.fun

        return final_joint_values, final_energy


if __name__ == "__main__":

    rospy.init_node('grasp_server_node')
    rospy.loginfo("starting grasp_server_node...")
    grasp_server = GraspServer()
    rospy.loginfo("Grasp Server Node ready for requests.")
    rospy.spin()