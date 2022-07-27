from calendar import c
import pybullet as p 
import numpy as np
import pybullet_data as pd
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
import os
import time
from utils.camera_util import *
from utils.object_util import * 
from tqdm import tqdm



def main():

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    model_path = "/home/chengjing/Desktop/pybullet-URDF-models/urdf_models/models"
    save_path = "/home/chengjing/Desktop/img_save_test"

    object_dict = {"book_1": {}, 
                "fork": {},
                "knife": {},
                "mug": {},
                "sugar_box": {}}

    obj_limits = {"x":(0.825, 1.025), "y":(0.25, 0.55), "z":(0, 0.05)}

    camera_look_at = [0.75, 0.5, 0.63] 


    num_camera_poses = 2

    num_worlds = 1

    base_pose = [0.6125, 0.5, 0.63]

    num_robot_config = 10

    seed = False
    
    seed_num = 0


    pybullet_world = PybulletWorldManager(num_worlds, object_dict, obj_limits, model_path)


    worlds = pybullet_world.get_world_list()

    pybullet_world.set_world_gravity()

    pybullet_world.load_default_world()

    for i, world in enumerate(tqdm(worlds, desc = "Total World")):

        pybullet_world.enable_real_time_simulation()

        pybullet_world.pybullet_set_world(world)

        world_save_path = os.path.join(save_path, world)

        try:
            os.mkdir(world_save_path)
        except FileExistsError:
            print("Folder already exists")
            exit()

        camera = PybulletCamera(num_camera_poses, camera_look_at)
        world_dict = {world: pybullet_world.world[world]}

        for j in tqdm(range(num_camera_poses), desc = f"World {i}", leave= False):
            cam_ = 'cam_' + str(j)
            cam_path = os.path.join(world_save_path, cam_) 
            os.mkdir(cam_path)

            color_img, depth_img, loaded_depth_img = camera.get_pose_img(j)

            print(camera.poses[j])

            camera.save_image(color_img, depth_img, cam_path, j)
            camera.save_pcd(j, color_img, loaded_depth_img, cam_path)
        
        camera_dict = camera.get_camera_dict()
        world_dict["camera"] = camera_dict

        dump_world(world_dict, world, world_save_path)

        pybullet_world.disable_real_time_simulation()

        panda = PandaArm(base_pose, num_robot_config, client, seed, seed_num)

        panda.label_generation()

        panda.save_data(world_save_path)

        panda.remove_panda()

        pybullet_world.pybullet_remove_world()

    
if __name__ == "__main__":
    main()
