import pybullet as p
import pybullet_data as pd
import numpy as np
from utils.general_util import *
from scipy.spatial.transform import Rotation as R
from PIL import Image
import h5py
import time

class PybulletCamera():
    def __init__(self, num_poses, look_at,  IMG_WIDTH = 512, IMG_HEIGHT= 512, FOV = 60, NEAR = 0.05, FAR = 5):
        self.num_poses = num_poses
        self.look_at = look_at
        self.focus = get_focus(IMG_WIDTH, fov = FOV)
        self.intrinsic = np.array([[self.focus, 0, (IMG_WIDTH + 1)/ 2], 
                                   [0, self.focus, (IMG_HEIGHT + 1) / 2],
                                   [0, 0, 1]])
        self.image_height = IMG_HEIGHT
        self.image_width = IMG_WIDTH
        self.far = FAR
        self.near = NEAR
        self.poses = self.pose_generation(num_poses)
        self.projection_matrx = get_projection_matrix(fov = FOV, near = NEAR, far = FAR)
        

        
    def get_camera_rpy(self, cam_translation):
        view_matrix = get_view_matrix(cam_translation, self.look_at, camera_up_vec=[0, 0, 1])
        view_matrix = np.array(view_matrix).reshape(4,4)
        gl2cv = R.from_euler("X", np.pi).as_matrix()

        rotation = view_matrix[:3,:3] @ gl2cv

        rpy = R.from_matrix(np.linalg.inv(rotation)).as_euler("xyz")
        
        return list(rpy)

    def get_view_matrix(self, i):
        view_matrix = get_view_matrix(self.poses[i,:3], self.look_at, camera_up_vec=[0, 1, 0])
        return  view_matrix


    def get_camera_image(self, view_matrix):
        _, _ , rgbImg, depthImg, _ = get_image(view_matrix, self.projection_matrx, width = self.image_width, height = self.image_height)
        depthImg = true_z_from_depth_buffer(depthImg, far = self.far, near = self.near)
        before = np.copy(depthImg)
        depthImg = self.process_depth_image(depthImg)
        loaded_depth_img = self.depth_image_from_load(depthImg)
        print("diff: ", np.mean(np.abs(loaded_depth_img - before)))
        return rgbImg, depthImg, before

    def process_depth_image(self, depth_img):
        mask = (depth_img == np.nan)
        depth_img = depth_image_range_conversion(depth_img, old_max=self.far, old_min=self.near, new_min=0, new_max=255)
        depth_img = np.around(depth_img).astype(np.uint8)
        depth_img[mask] = 0
        return depth_img
    
    def depth_image_from_load(self, depth_img):
        # Imitate behavior of reading and converting depth image from file
        return depth_image_range_conversion(depth_img, self.near, self.far)

    def get_pose_img(self, i):
        view_matrix = self.get_view_matrix(i)
        return self.get_camera_image(view_matrix)


    def pose_gen(self, theta , phi, radius, x, y, z):

        # Parametric coordinate generation
        theta = np.random.rand() * 2 * theta + (-theta)  #  ± 15 degree default
        phi  = np.random.rand() * 2 * phi + (-phi)
        radius = radius + (np.random.rand() * 0.2 + -0.1) # ± 10 cm radius

        cam_x  = radius * np.cos(theta) * np.sin(phi) + x
        cam_y = radius * np.sin(theta) * np.sin(phi) + y 
        cam_z = radius * np.cos(phi) + z


        cam_pose = np.array([cam_x, cam_y, cam_z]) 

        rpy = self.get_camera_rpy(cam_pose)


        return np.array([cam_x, cam_y, cam_z, rpy[0], rpy[1], rpy[2]])
    

    def pose_generation(self, num_poses,theta = np.pi * (1 / 3), phi = np.pi * (1 / 3) , radius = 1.65, x = 0.75, y = 0.4, z= 1.07):
        poses = np.zeros((num_poses, 6))

        for i in range(num_poses):
            poses[i,:] = self.pose_gen(theta = theta, phi = phi, radius = radius,  x = x, y = y, z = z)
        self.poses = poses
        
        return poses
    
    def save_image(self, color_img, depth_img, path, cam_id):

        cam_id = 'cam_' + str(cam_id)

        color_img = Image.fromarray(color_img)
        color_path = os.path.join(path, cam_id + "_color.png")
        color_img.save(color_path)

        depth_img = Image.fromarray(depth_img)
        depth_path = os.path.join(path, cam_id + "_depth.png")
        depth_img.save(depth_path)
        

    
    def get_camera_dict(self):
        cam_dict = {}
        cam_list = []
        for i in range(self.num_poses):
            pose = self.poses[i]
            cam_list.append({"xyz": pose[:3].tolist(), "rpy": pose[3:].tolist()})
        cam_dict["pose"] = cam_list
        cam_dict["intrinsic"] = self.intrinsic.tolist()
        cam_dict["focus"] = self.focus
        cam_dict["far"] = self.far
        cam_dict["near"] = self.near
        cam_dict["image_height"] = self.image_height
        cam_dict["image_width"] = self.image_width
        return cam_dict



    def save_pcd(self, cam_pose, color_img, depth_img, path):
        view_matrix = np.array(self.get_view_matrix(cam_pose)).reshape(4,4).T

        gl2cv = R.from_euler("X", np.pi).as_matrix()

        view_matrix = np.linalg.inv(view_matrix)

        view_matrix[:3,:3] = view_matrix[:3,:3] @ gl2cv

        print("get_pose:", view_matrix[:3,3])
        print("set_pose:", self.poses[cam_pose,:3])


        print(self.intrinsic)

        extrinsic = view_matrix

        cam = get_camera(extrin = extrinsic, height=self.image_height, width=self.image_width, f = self.focus)

        rgbd = buffer_to_rgbd(color_img, depth_img)

        pcd = get_pcd(cam, rgbd)

        xyz = np.array(pcd.points)
        color = np.array(pcd.colors)

        pc_path = os.path.join(path, "pc.h5")
        color_path = os.path.join(path, "color.h5")

        with h5py.File(pc_path, 'w') as hf:
            hf.create_dataset("pointcloud", data=xyz)
    
        with h5py.File(color_path, 'w') as hf:
            hf.create_dataset("color", data=color)

