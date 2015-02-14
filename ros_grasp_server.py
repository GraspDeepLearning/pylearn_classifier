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

        #conv_model_name = "processed_2.0m_7vc_barrett_18_grasp_types_5_layer_170x170_1_12_12_41"
        #conv_model_name = "processed_out_condensed_5_layer_170x170_2_6_16_55"
        #conv_model_name = "random_5_layer_170x170_2_6_16_43"
        self.conv_model_name = "processed_sweet_guava_5_layer_72x72_2_13_10_29"

        try:
            rospy.wait_for_service('/uvd_to_xyz', 6)
        except rospy.ServiceException as exc:
            rospy.logerr("uvd_to_xyz service never came up")
            assert False

        self.uvd_to_xyz_proxy = rospy.ServiceProxy('uvd_to_xyz', UVDTOXYZ)

        f = open("grasp_priors_list.pkl")
        self.grasp_priors_list = pickle.load(f)

        conv_model_filepath = paths.MODEL_DIR + self.conv_model_name + "/cnn_model.pkl"

        dataset_file = str(int(round(time.time() * 1000)))

        input_dset, raw_rgbd_filepath = init_rgbd_file(dataset_file)
        self.input_dset = input_dset
        self.input_dset.create_dataset("rgbd", (1, 480, 640, 4))

        save_dset, save_filepath = init_save_file(dataset_file, self.conv_model_name)
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

        response = CalculateGraspsServiceResponse()
        response.heatmaps = self.save_dset['rescaled_heatmaps'][0].flatten()
        response.heatmap_dims = self.save_dset['rescaled_heatmaps'][0].shape
        response.model_name = self.conv_model_name

        return response

        # grasps = self.pipeline._pipeline_stages[-1].grasps
        #
        # grasp_msgs = CalculateGraspsServiceResponse()
        #
        # for grasp in grasps:
        #     grasp_energy, grasp_type,  palm_index, argmax_v, argmax_u, x_border, y_border, heatmaps = grasp
        #     d = rgbd[argmax_v+x_border, argmax_u+y_border, 3]
        #
        #     rospy.loginfo("u: " + str(argmax_u))
        #     rospy.loginfo("v: " + str(argmax_v))
        #     rospy.loginfo("d: " + str(d))
        #
        #     resp = self.uvd_to_xyz_proxy(argmax_u+x_border, argmax_v+y_border, d)
        #     x = resp.x
        #     y = resp.y
        #     z = resp.z
        #
        #     joint_values = self.grasp_priors_list.get_grasp_prior(grasp_type).joint_values
        #     wrist_roll = self.grasp_priors_list.get_grasp_prior(grasp_type).wrist_roll
        #
        #     quat = tf.transformations.quaternion_from_euler(0, math.pi/2.0, wrist_roll, axes='szyx')
        #     #quat = tf.transformations.quaternion_from_euler(0, 0,  wrist_roll, axes='szyx')
        #
        #
        #     grasp_msg = Grasp()
        #     grasp_msg.pose.position.x = x
        #     grasp_msg.pose.position.y = y
        #     grasp_msg.pose.position.z = z
        #     grasp_msg.pose.orientation.x = quat[0]
        #     grasp_msg.pose.orientation.y = quat[1]
        #     grasp_msg.pose.orientation.z = quat[2]
        #     grasp_msg.pose.orientation.w = quat[3]
        #
        #     jv = JointState()
        #     jv.name = ["bhand/finger_1/prox_joint",
        #                "bhand/finger_1/med_joint",
        #                "bhand/finger_1/dist_joint",
        #                "bhand/finger_2/prox_joint"
        #                "bhand/finger_2/med_joint",
        #                "bhand/finger_2/dist_joint",
        #                "bhand/finger_3/med_joint",
        #                "bhand/finger_3/dist_joint"]
        #     jv.position = joint_values
        #     grasp_msg.joint_values = jv
        #     grasp_msg.grasp_energy = grasp_energy
        #
        #     grasp_msg.grasp_type = grasp_type
        #
        #     grasp_msgs.grasps.append(grasp_msg)
        #
        # rospy.loginfo("finished")
        #
        # refined_grasp_msgs = CalculateGraspsServiceResponse()
        # for i in range(len(grasps)):
        #     grasp_energy, grasp_type,  palm_index, argmax_v, argmax_u, x_border, y_border, heatmaps = grasps[i]
        #     d = rgbd[argmax_v + x_border, argmax_u + y_border, 3]
        #     grasp_msg = grasp_msgs.grasps[i]
        #     grasp_refiner = RefineGrasps(grasp_msg=grasp_msg,
        #              palm_uvd=(argmax_u, argmax_v, d),
        #              heatmaps=heatmaps,
        #              x_border=x_border,
        #              y_border=y_border)
        #
        #     final_joint_values, final_energy = grasp_refiner.run()
        #     grasp_msg.joint_values.position = final_joint_values
        #     refined_grasp_msgs.grasps.append(grasp_msg)
        #
        # refined_grasp_msgs.heatmaps = self.save_dset['independent_x_priors'][0].flatten()
        # refined_grasp_msgs.heatmap_dims = self.save_dset['independent_x_priors'][0].shape
        # return refined_grasp_msgs


from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
import image_geometry
from sensor_msgs.msg import CameraInfo
from scipy.optimize import minimize
import tf
import tf_conversions

class RefineGrasps():

    def __init__(self, grasp_msg, palm_uvd, heatmaps, x_border, y_border, camera_param_topic='/camera/rgb/camera_info'):
        self.palm_uvd = palm_uvd
        self.grasp_msg = grasp_msg
        self.heatmaps = heatmaps
        self.x_border= x_border
        self.y_border = y_border

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

        self.robot = URDF.from_parameter_server('demo_hand_robot_description')
        self.tree = kdl_tree_from_urdf_model(self.robot)

        camera_info = rospy.wait_for_message(camera_param_topic, CameraInfo)
        self.pinhole_model = image_geometry.PinholeCameraModel()
        self.pinhole_model.fromCameraInfo(camera_info)

        base_link = 'bhand/bhand_palm_link'
        self.f_1_kdl_kin = KDLKinematics(self.robot, base_link, 'bhand/finger_1/dist_link')
        self.f_2_kdl_kin = KDLKinematics(self.robot, base_link, 'bhand/finger_2/dist_link')
        self.f_3_kdl_kin = KDLKinematics(self.robot, base_link, 'bhand/finger_3/dist_link')

    def xyz_to_uv(self, position):
        return self.pinhole_model.project3dToPixel(position)

    def cost_function(self, joint_values):
        f1_joint_angles = joint_values[0:3]
        f2_joint_angles = joint_values[3:6]
        f3_joint_angles = joint_values[6:8]

        #these are 4x4 matrices
        f_1_pose = self.f_1_kdl_kin.forward(f1_joint_angles)
        f_2_pose = self.f_2_kdl_kin.forward(f2_joint_angles)
        f_3_pose = self.f_3_kdl_kin.forward(f3_joint_angles)

        palm_pose = tf_conversions.toMatrix(tf_conversions.fromMsg(self.grasp_msg.pose))
        f_1_pose_dot_palm = np.dot(palm_pose, f_1_pose)
        f_2_pose_dot_palm = np.dot(palm_pose, f_2_pose)
        f_3_pose_dot_palm = np.dot(palm_pose, f_3_pose)

        f1_xyz = tf.transformations.translation_from_matrix(f_1_pose_dot_palm)
        f2_xyz = tf.transformations.translation_from_matrix(f_2_pose_dot_palm)
        f3_xyz = tf.transformations.translation_from_matrix(f_3_pose_dot_palm)

        f1_u, f1_v = self.xyz_to_uv(f1_xyz)
        f2_u, f2_v = self.xyz_to_uv(f2_xyz)
        f3_u, f3_v = self.xyz_to_uv(f3_xyz)
        p_u, p_v, _ = self.palm_uvd

        palm_energy = self.heatmaps[0, p_v - self.x_border, p_u - self.y_border]

        f_1_energy = -100
        if f1_v < self.heatmaps.shape[1] and f1_u < self.heatmaps.shape[2]:
            f_1_energy = self.heatmaps[1, f1_v - self.x_border, f1_u - self.y_border]

        f_2_energy = -100
        if f2_v < self.heatmaps.shape[1] and f2_u < self.heatmaps.shape[2]:
            f_2_energy = self.heatmaps[2, f2_v - self.x_border, f2_u - self.y_border]

        f_3_energy = -100
        if f3_v < self.heatmaps.shape[1] and f3_u < self.heatmaps.shape[2]:
            f_3_energy = self.heatmaps[3, f3_v - self.x_border, f3_u - self.y_border]

        return -palm_energy - f_1_energy - f_2_energy - f_3_energy

    def run(self):
        # add a constraint for each joint in order to ensure they stay within
        # feasible joint range.
        constraints = []

        for i in range(len(self.grasp_msg.joint_values.position)):

            #joint values must be below max joint limit
            constraints.append({'type': 'ineq', 'fun': lambda x: np.array([x[i] - self.max_joint_limits[i]])})

            #joint values must be above min joint limit
            constraints.append({'type': 'ineq', 'fun': lambda x: np.array([x[i] - self.min_joint_limits[i]])})

        # the proximal joint for finger 1 and finger 2 must always be equal
        constraints.append({'type': 'ineq', 'fun': lambda x: np.array([x[0] - x[3]])})

        result = minimize(self.cost_function, self.grasp_msg.joint_values.position, constraints=constraints,)

        final_joint_values = result.x
        final_energy = result.fun

        return final_joint_values, final_energy


if __name__ == "__main__":

    rospy.init_node('grasp_server_node')
    rospy.loginfo("starting grasp_server_node...")
    grasp_server = GraspServer()
    rospy.loginfo("Grasp Server Node ready for requests.")
    rospy.spin()