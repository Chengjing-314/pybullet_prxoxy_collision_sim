from distutils.extension import Extension
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from utils.moveit_util import *
from sensor_msgs.msg import PointCloud2
import threading
import time


def pc_thread(topic, pc_msg):
    global pose_switch_flag 
    pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
    rospy.sleep(0.1) # allow time for the publisher to connect
    rate = rospy.Rate(10)
    while not pose_switch_flag:
        pub.publish(pc_msg)
        rate.sleep()
    print("pc_thread exit")



def main():
    
    #General Parameters for World and Camera
    num_world = 1
    num_cam = 10
    
    #Parameter for PC, filter out points in z axis > distance
    filter = True
    distance = 1.4
    
    
    global pose_switch_flag 
    rospy.init_node("fake_sensor")
    path = "/home/chengjing/Desktop/cam_test"
    
    world = 0
    cam = 0

    pose_switch_flag = True
    cfg_path = os.path.join(path, "world_" + str(world), "robot_config.pt")
    cfgs = torch.load(cfg_path)
    gt_label = torch.load(os.path.join(path, "world_" + str(world), "collision_label.pt"))
    octomap_label = torch.load(os.path.join(path, "world_" + str(world), 'cam_' + str(cam), "pc_collision_label.pt"))
    
    mask = ( torch.logical_and((gt_label== 1), (gt_label != octomap_label)))
    
    print("total: ", torch.sum(mask))
    
    print("sum_total_different", torch.sum(gt_label != octomap_label))

    
    cfgs = cfgs[mask]
    print("world: ", world)
    pose_switch_flag = False
    pc = fake_sensor(world, cam, path, filter, distance)
    time.sleep(1) # Give time for frame publisher to connect and publish
    pc_pub_thread = threading.Thread(target=pc_thread, args=("/camera/depth/points", pc))
    pc_pub_thread.start()
    time.sleep(0.5) # Give time for pc to publish
    moveit = MoveitDataGen(cfgs)
    moveit.cfg_visualization(cfgs)
    moveit.clear_octomap() # clear out octomap noise
    time.sleep(3) # Wait for octomap to be cleared and pc reload
    cam_path = os.path.join(path, "world_" + str(world), "cam_" + str(cam))
    data_path = os.path.join(cam_path, "pc_collision_label.pt")
    moveit.dump_data(data_path)
    pose_switch_flag = True
    pc_pub_thread.join()
    moveit.clear_octomap()
    time.sleep(0.5)


if __name__ == "__main__":
    main()
            