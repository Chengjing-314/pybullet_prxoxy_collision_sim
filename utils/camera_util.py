import numpy as np
from utils.general_util import *
from scipy.spatial.transform import Rotation as R
import imageio
import h5py


class PybulletCamera():
    def __init__(self, num_poses, look_at, multipler = 10, img_width = 512, img_height= 512, fov = 60, near = 0.05, far = 5):
        self.num_poses = num_poses
        self.look_at = look_at
        self.projection_matrx = get_projection_matrix(fov = fov, near = near, far = far)
        self.focus = get_focus(img_height, fov)
        self.intrinsic = np.array([[self.focus, 0, (img_width + 1)/ 2], 
                                   [0, self.focus, (img_height + 1) / 2],
                                   [0, 0, 1]])
        self.image_height = img_height
        self.image_width = img_width
        self.far = far
        self.near = near
        self.multipler = multipler
        
        
    def get_view_matrix(self, i):
        view_matrix = get_view_matrix(self.poses[i,:3], self.look_at, camera_up_vec=[0, 0, 1])
        return  view_matrix


    def get_camera_image(self, view_matrix):
        _, _ , rgbImg, depthImg, _ = get_image(view_matrix, self.projection_matrx, width = self.image_width, height = self.image_height)
        depthImg = true_z_from_depth_buffer(depthImg, far = self.far, near = self.near)
        # depthImg = self.process_depth_image(depthImg) 
        depthImg = self.process_depth_image_range_conversion(depthImg)
        # The difference is between original and saved depth image is less than 5*10^-9
        loaded_depth_img = self.depth_image_from_load(depthImg) 
        return rgbImg, depthImg, loaded_depth_img


    def process_depth_image(self, depth_img):
        mask = (depth_img == np.nan)
        depth_img = depth_img * 10000 
        depth_img[mask] = 0
        depth_img = depth_img.astype(np.uint16)
        return depth_img
    
    def process_depth_image_range_conversion(self, depth_img):
        uint16_min, uint16_max  = 0, 2**16 - 1
        mask = (depth_img == np.nan)
        depth_img = depth_image_range_conversion(depth_img, uint16_min, uint16_max, self.near * self.multipler, self.far).astype(np.uint16)
        depth_img[mask] = 0
        return depth_img
    
    def depth_image_from_load(self, depth_img):
        # Imitate behavior of reading and converting depth image from file
        uint16_min, uint16_max  = 0, 2**16 - 1
        depth_img = depth_img.astype(np.float32)
        depth_img[depth_img == 0] = np.nan
        # return  depth_img / 10000
        depth_img = depth_image_range_conversion(depth_img, self.near * self.multipler, self.far, uint16_min, uint16_max)
        return depth_img

    def get_pose_img(self, i):
        view_matrix = self.get_view_matrix(i)
        return self.get_camera_image(view_matrix)


    def pose_gen(self, theta , phi, radius, x, y, z, theta_var = np.pi / 6, phi_var = np.pi / 12, radius_var = 0.05, theta_offset = 0.25):

        # Parametric coordinate generation
        temp = theta
        theta = (np.random.rand() * 2  - 1) * theta_var + theta
        theta += theta_offset if theta >= temp else -theta_offset
        phi  = (np.random.rand() * 2  - 1) * phi_var + phi
        radius = (np.random.rand() * 1 + -1) * radius_var + radius

        # print("theta: ", theta, "phi: ", phi, "radius: ", radius)

        cam_x  = radius  * np.cos(phi) * np.cos(theta) + x
        cam_y = radius  * np.cos(phi) * np.sin(theta) + y 
        cam_z = radius * np.sin(phi) + z


        cam_pose = np.array([cam_x, cam_y, cam_z]) 

        rpy = self.get_camera_rpy(cam_pose)


        return np.array([cam_x, cam_y, cam_z, rpy[0], rpy[1], rpy[2]])

    

    def pose_generation(self, num_poses,theta = np.pi * (1 / 3), phi = np.pi * (1 / 3) , radius = 1.65, x = 0.75, y = 0.4, z= 1.07, 
    theta_var = np.pi / 6, phi_var = np.pi / 12, radius_var = 0.05, theta_offset = 0.25):

        poses = np.zeros((num_poses, 6))

        for i in range(num_poses):
            poses[i,:] = self.pose_gen(theta = theta, phi = phi, radius = radius,  x = x, y = y, z = z, 
                                        theta_var=theta_var, phi_var=phi_var, radius_var=radius_var)
        self.poses = poses
        
        return poses
    
    def save_image(self, color_img, depth_img, path, cam_id):

        cam_id = 'cam_' + str(cam_id)

        color_path = os.path.join(path, cam_id + "_color.png")
        imageio.imwrite(color_path, color_img)

        depth_path = os.path.join(path, cam_id + "_depth.png")
        imageio.imwrite(depth_path, depth_img)
        

    
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
        cam_dict["multipler"] = self.multipler
        return cam_dict



    def get_camera_rpy(self, cam_translation):
        view_matrix = get_view_matrix(cam_translation, self.look_at, camera_up_vec=[0, 0, 1])
        view_matrix = np.array(view_matrix).reshape(4,4).T # OpenGL is column major :(
        # view_matrix = np.linalg.inv(view_matrix)
        gl2cv = R.from_euler("X", np.pi).as_matrix()

        rotation = np.linalg.inv(view_matrix[:3,:3]) @ gl2cv


        rpy = R.from_matrix(rotation).as_euler("xyz")
        
        return list(rpy)


    def save_pcd(self, color_img, depth_img, path):

        # view_matrix = np.array(self.get_view_matrix(cam_pose)).reshape(4,4).T

        # gl2cv = R.from_euler("X", np.pi).as_matrix()

        # view_matrix = np.linalg.inv(view_matrix)

        # view_matrix[:3,:3] = view_matrix[:3,:3] @ gl2cv

        # # view_matrix[:3,3] = self.poses[cam_pose][:3]

        # # print("get_pose:", view_matrix[:3,3])
        # # print("set_pose:", self.poses[cam_pose,:3])

        # # print(self.intrinsic)

        # extrinsic = view_matrix

        cam = get_camera(extrin = np.eye(4), height=self.image_height, width=self.image_width, f = self.focus)
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

