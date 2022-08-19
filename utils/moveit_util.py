import numpy as np
from scipy.spatial.transform import Rotation as R
from moveit_msgs.msg import RobotState, DisplayTrajectory, RobotTrajectory
from moveit_msgs.srv import GetStateValidityRequest, GetStateValidity
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import Empty
from utils.data_reader_util import *
import std_msgs.msg
import rospy 
import sys
import moveit_commander
import tf2_ros as tf2
from tqdm import tqdm
import torch
from sensor_msgs.point_cloud2 import create_cloud_xyz32
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import os
import h5py
import time

pi = np.pi


class MoveitDataGen(object):
    def __init__(self, cfgs):
        super().__init__()

        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.joint_name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 
                            'panda_joint6', 'panda_joint7']
        self.cfgs = cfgs 
        self.labels = torch.zeros(self.cfgs.size(dim=0), dtype=torch.float)
        self.DOF = 7
        self.sv_srv = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.display_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=20)


    def data_generation(self):
  
        rs = RobotState()
        rs.joint_state.name = self.joint_name
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = rs
        gsvr.group_name = self.group_name 
        time.sleep(0.1)
        for i, cfg in enumerate(tqdm(self.cfgs)):
            rs.joint_state.position = cfg
            result =self.sv_srv.call(gsvr)

            in_collision = not result.valid
            self.labels[i]  = 1 if in_collision else -1 # FIXME: change to 0?
            
        print("{} in collision, {} collision free".format(torch.sum(self.labels == 1), torch.sum(self.labels == -1)))
        return self.labels
    
    def dump_data(self, path):
        torch.save(self.labels, path)

        
    
    def cfg_visualization(self, cfgs): # Test Purpose Only
        for i, cfg in enumerate(cfgs):
            input('------------------Press Enter To Continue------------------')
            # if labels[i] == 1:
            #     print("In collision")
            # else:
            #     print("No collision")
            self.trajectory_visual(cfg)
            



    def cfg_visualize_all(self, cfgs, labels):
        joint_traj = JointTrajectory()
        joint_traj.header = std_msgs.msg.Header()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.joint_names = self.joint_name

        start = [0, -pi/4, 0, -pi/2, 0, pi/3, 0]

        for i, cfg in enumerate(cfgs):
            joint_pts = JointTrajectoryPoint()
            joint_pts.positions = cfg.numpy().tolist()
            joint_traj.points.append(joint_pts)

        joint_pts = JointTrajectoryPoint()
        joint_pts.positions = start
        joint_traj.points.append(joint_pts)
        

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj
        disp_traj = DisplayTrajectory()
        disp_traj.model_id = 'panda' 
        disp_traj.trajectory.append(robot_traj)
        disp_traj.trajectory_start.joint_state.position = start
        disp_traj.trajectory_start.joint_state.name = joint_traj.joint_names

        labels = labels.numpy().tolist()

        mask = ["collision" if i == 1 else "free" for i in labels]

        print(f'Collision Labels:{mask}')

        rospy.sleep(1) # let the subscriber connect 

        self.display_pub.publish(disp_traj)



    def trajectory_visual(self, cfg,  start = [0, -pi/4, 0, -pi/2, 0, pi/3, 0]): 
        joint_traj = JointTrajectory()
        joint_traj.header = std_msgs.msg.Header()
        joint_traj.header.stamp = rospy.Time.now()
        joint_traj.joint_names = self.joint_name

        joint_pts = JointTrajectoryPoint()
        joint_pts.positions = cfg.numpy().tolist()

        joint_traj.points.append(joint_pts)

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = joint_traj
        disp_traj = DisplayTrajectory()
        disp_traj.model_id = 'panda' 
        disp_traj.trajectory.append(robot_traj)
        disp_traj.trajectory_start.joint_state.position = start
        disp_traj.trajectory_start.joint_state.name = joint_traj.joint_names
        self.display_pub.publish(disp_traj)


        print(f"set state:\n {joint_pts.positions}")
        
    def clear_octomap(self):
        rospy.wait_for_service('/clear_octomap')
        clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)
        clear_octomap()






def fake_sensor(world, cam, path, filter = False, distance = 1.7): # cam and world are int, not str

    cam_name= "cam_" + str(cam)
    world_name = "world_" + str(world)

    # pointcloud_pub = rospy.Publisher("/camera/depth/points", PointCloud2)

    world_path = os.path.join(path, world_name)
    

    world_dict_path = os.path.join(world_path, world_name + ".json")

    wolrd_dict = get_world_dict(world_dict_path)

    translation, rotation = get_camera_pose(wolrd_dict, cam)

    current_pose = translation + rotation

    # Notify the Frame Publisher that there is a pose change

    pose_pub = rospy.Publisher("/pose_update", Float64MultiArray, queue_size=10)

    pose_msg = Float64MultiArray()

    pose_msg.data = current_pose

    rospy.sleep(0.1) # allow time for the publisher to connect

    pose_pub.publish(pose_msg)

    # Transform the point cloud to world frame

    # tf_buffer = tf2.Buffer()
    # lr = tf2.TransformListener(tf_buffer)

    # trans = tf_buffer.lookup_transform("world","camera_frame",rospy.Time(0), rospy.Duration(4.0))

    rospy.sleep(0.1) # wait for frame to be published
    
    header = Header()
    header.stamp = rospy.Time.now()
    # header.frame_id = "camera_frame"
    header.frame_id = "camera_frame"

    cam_path = os.path.join(world_path, cam_name)
    pc_path = os.path.join(cam_path, "pc.h5")

    with h5py.File(pc_path) as f:
        xyz = f["pointcloud"][:]
        
    if filter:
        mask = np.where(xyz[:,2] > distance)[0]
        xyz = np.delete(xyz, mask, axis = 0)
                

    pcd_msg = create_cloud_xyz32(header, xyz)

    # pcd_msg = do_transform_cloud(pcd_msg, trans)

    # pcd_msg.header.frame_id = "camera_frame"
    # pcd_msg.header.frame_id = "world"

    return pcd_msg