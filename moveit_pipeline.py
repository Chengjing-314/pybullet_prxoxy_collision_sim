from utils.moveit_util import *
from sensor_msgs.msg import PointCloud2

path = "/home/chengjing/Desktop/img_save_test"


rospy.init_node("fake_sensor")

pc = fake_sensor(0,0,path)

pc_pub = rospy.Publisher("/camera/depth/points", PointCloud2, queue_size=1)

rospy.sleep(0.1)

rate = rospy.Rate(10)

while not rospy.is_shutdown():
    pc_pub.publish(pc)
    rate.sleep()