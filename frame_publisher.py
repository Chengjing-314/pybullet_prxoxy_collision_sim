import rospy
import tf2_ros
import tf2_msgs.msg
import geometry_msgs.msg
import numpy as np
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R



class FramePublisher():

    def __init__(self):
        self.camera_frame_TF = None
        self.optical_refrence_frame_TF = None
        self.current_pose = None

        self.br = tf2_ros.TransformBroadcaster()
        self.pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)


        while not rospy.is_shutdown():
            self.pos_update_sub = rospy.Subscriber("/pose_update", Float64MultiArray, self.pos_update_callback)
            rospy.sleep(0.2)
            if self.current_pose:
                self.camera_frame_TF = self.cam_frame_builder(self.current_pose)
                self.pub_tf.publish(self.camera_frame_TF)
                self.optical_refrence_frame_TF = self.optical_reference_frame_builder()
                self.pub_tf.publish(self.optical_refrence_frame_TF)
            else:
                continue

    
    def pos_update_callback(self, msg):
        self.current_pose = msg.data    
        
    

    def cam_frame_builder(self, pose):
        print(pose)
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = "world"
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "camera_frame"


        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = pose[2]


        r = pose[3:]
        r = R.from_euler('xyz', r).as_quat()

        t.transform.rotation.x = r[0]
        t.transform.rotation.y = r[1]
        t.transform.rotation.z = r[2]
        t.transform.rotation.w = r[3]

        tfm = tf2_msgs.msg.TFMessage([t])
        return tfm
    

    def optical_reference_frame_builder(self):
        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = "camera_frame"
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = "optical_reference_frame"

        t.transform.translation.x = 0
        t.transform.translation.y = 0
        t.transform.translation.z = 0
        pi = np.pi

        r = [pi/2, 0, pi/2] # Gazebo camera frame to rviz camera frame 

        r = R.from_euler('XYZ', r).as_quat()

        t.transform.rotation.x = r[0]
        t.transform.rotation.y = r[1]
        t.transform.rotation.z = r[2]
        t.transform.rotation.w = r[3]

        tfm = tf2_msgs.msg.TFMessage([t])
        return tfm



if __name__ == '__main__':
    rospy.init_node('cam_frame_publisher')
    tfb = FramePublisher()

    rospy.spin()