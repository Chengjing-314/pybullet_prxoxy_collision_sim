import pybullet as p
import pybullet_data as pd
import numpy as np
from utils.general_util import *
from scipy.spatial.transform import Rotation as R

class PybulletCamera():
    def __init__(self, num_poses, look_at, IMG_LEN = 512, IMG_WIDTH = 512, FOV = 60, NEAR = 0.05, FAR = 3):
        self.num_poses = num_poses
        self.poses = None
        self.look_at = look_at
        self.projection_matrx = get_projection_matrix(fov = FOV, near = NEAR, far = FAR)
        self.focus = get_focus(IMG_LEN = IMG_LEN, fov = FOV)
        self.intrinsic = np.array([[f, 0, (IMG_LEN-1) / 2], 
                                   [0, f, (IMG_WIDTH-1) / 2],
                                   [0, 0, 1]])
        self.image_len = IMG_LEN
        self.image_width = IMG_WIDTH
        self.far = FAR
        self.near = NEAR
        
    def get_camera_rpy(self, cam_translation):
        view_matrix = get_view_matrix(cam_translation, self.look_at, camera_up_vec=[0, 0, 1])
        view_matrix = np.array(view_matrix).reshape(4,4)
        gl2cv = R.from_euler("X", np.pi/2).as_matrix()
        rotation = np.linalg.inv(view_matrix)[:3,:3]

        rpy = R.form_matrix(gl2cv @ rotation).as_euler("xyz")
        
        return list(rpy)

    def get_view_matrix(self, i):
        return  get_view_matrix(self.cam_poses[i], self.look_at, camera_up_vec=[0, 0, 1])


    def get_camera_image(self, view_matrix):
        _, _ , rgbImg, depthImg, _ = get_image(view_matrix, self.projection_matrx, width = self.image_width, height = self.image_len)
        depthImg = true_z_from_depth_buffer(depthImg, self.far, self.near)
        return rgbImg, depthImg

    def pose_gen(self, theta , phi, radius, x, y, z):

        # Parametric coordinate generation
        theta = np.random.rand() * 2 * theta + (-theta)  #  ± 15 degree default
        phi  = np.random.rand() * 2 * phi + (-phi)
        radius = radius + (np.random.rand() * 0.2 + -0.1) # ± 10 cm radius

        cam_x  = radius * np.cos(theta) * np.sin(phi) + x
        cam_y = radius * np.sin(theta) * np.sin(phi) + y 
        cam_z = radius * np.cos(phi) + z


        cam_pose = np.array([cam_x, cam_y, cam_z]) 
        target = np.array([x, y, z])

        rpy = self.get_camera_rpy(cam_pose, target)


        return np.array([cam_x, cam_y, cam_z, rpy[0], rpy[1], rpy[2]])
    

    def pose_generation(self, num_poses,theta = np.pi * (1 / 3), phi = np.pi * (1 / 3) , radius = 1.65, x = 0.75, y = 0.4, z= 1.07):
        poses = np.zeros((num_poses, 6))

        for i in range(num_poses):
            poses[i,:] = self.pose_gen(theta = theta, phi = phi, radius = radius,  x = x, y = y, z = z)
        self.poses = poses
        
        return poses