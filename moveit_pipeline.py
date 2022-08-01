from utils.moveit_util import *
from sensor_msgs.msg import PointCloud2
import threading
import time



def pc_pub_thread(topic, pc_msg):
    global pose_switch_flag 
    pub = rospy.Publisher(topic, PointCloud2, queue_size=1)
    rospy.sleep(0.1) # allow time for the publisher to connect
    rate = rospy.Rate(10)
    while not pose_switch_flag:
        pub.publish(pc_msg)
        rate.sleep()



def main():
    global pose_switch_flag 
    rospy.init_node("fake_sensor")
    path = "/home/chengjing/Desktop/img_save_test"

    num_world = 1
    num_cam = 0

    pose_switch_flag = True
    for world in range(num_world):
        cfg_path = os.path.join(path, "world_" + str(world), "robot_config.pt")
        cfgs = torch.load(cfg_path)
        for cam in range(num_cam):
            pose_switch_flag = False
            pc = fake_sensor(world, cam, path)
            pc_pub_thread = threading.Thread(target=pc_pub_thread, args=("/camera/depth/points", pc))
            pc_pub_thread.start()
            moveit = MoveitDataGen(cfgs)
            moveit.data_generation()
            cam_path = os.path.join(path, "world_" + str(world), "cam_" + str(cam))
            moveit.dump_data(cam_path)
            pose_switch_flag = True
            pc_pub_thread.join()
            time.sleep(0.5)


if __name__ == "__main__":
    main()
            