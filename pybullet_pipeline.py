import pybullet as p 
import pybullet_data as pd
import os
from utils.camera_util import *
from utils.object_util import * 
from tqdm import tqdm

def main():

    pi = np.pi

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())

    # Overall Parameters
    num_camera_poses = 10 # Number of camera poses per world
    num_worlds = 1 # Number of worlds
    max_objects_per_world = 8 # Maximum number of objects per world
    model_path = "/home/chengjing/Desktop/pybullet-URDF-models/urdf_models/models"
    save_path = "/home/chengjing/Desktop/save_new_ratio"

    # Object Parameters
    object_dict = {
                "book_1": {}, 
                "fork": {},
                "knife": {},
                "spoon": {},
                "mug": {},
                "sugar_box": {},
                "lipton_tea":{},
                "plastic_apple":{},
                "plastic_banana":{},
                "potato_chip_1":{},
                "scissors":{},
                "poker_1":{},
                "remote_controller_1": {},
                "plate": {},
                "glue_1": {}
                }

    obj_limits = {"x":(0.665, 0.965), "y":(0.2, 0.8), "z": 0.1} # Object spawning limits
    obj_z_offset = 0.63 # offset of the table
   
    
    
    # Camera Parameters
    camera_look_at = [0.815, 0.5, 0.63] # Camera look at point
    camera_phi =  pi *  4 / 9  # Camera phi angle
    camera_theta = pi   # Camera theta angle
    camera_radius = 0.85 # Camera radius
    camera_x, camera_y, camera_z = camera_look_at[0], camera_look_at[1], camera_look_at[2]
    camera_multipler = 10
    camera_phi_var, camera_theta_var, camera_radius_var = pi / 12, pi / 6, 0.05 # Camera phi theta and radius variance


    # Panda arm parameter
    area_of_interest = {"x":(0.665, 0.965), "y":(0.3, 0.6), "z": (0.63, 0.67)} # Area of interest for camera
    ratio = 0.5 # Rough ratio of in AOI and out of AOI
    panda_base_pose = [0.315, 0.5, 0.63] 
    num_robot_config = 10000
    seed = False
    seed_num = 0
    
    np.random.seed(2023)

    # temp_arm = p.loadURDF("franka_panda/panda.urdf", panda_base_pose)

    # Initialize world
    pybullet_world = PybulletWorldManager(num_worlds, object_dict, obj_limits, model_path, obj_z_offset, max_objects_per_world)

    # Get the world list for iteration
    worlds = pybullet_world.get_world_list()
    pybullet_world.set_world_gravity()

    # Load default world with plane and table
    pybullet_world.load_default_world()

    for i, world in enumerate(tqdm(worlds, desc = "Total World")):
        # Enable real time sim for object to drop
        pybullet_world.enable_real_time_simulation()

        # Set current world to generated world 
        pybullet_world.pybullet_set_world(world)

        world_save_path = os.path.join(save_path, world)

        try:
            os.mkdir(world_save_path)
        except FileExistsError: # Exit if the folder already exists
            print("Folder already exists")
            exit()

        camera = PybulletCamera(num_camera_poses, camera_look_at, camera_multipler) # Initialize camera
        camera.pose_generation(camera.num_poses, camera_theta, camera_phi, camera_radius, camera_x, camera_y, camera_z
                              ,camera_theta_var, camera_phi_var, camera_radius_var) # Pose generation

        world_dict = {world: pybullet_world.world[world]}

        for j in tqdm(range(num_camera_poses), desc = f"World {i}", leave= False):
            cam_ = 'cam_' + str(j)
            cam_path = os.path.join(world_save_path, cam_) 
            os.mkdir(cam_path)

            color_img, depth_img, loaded_depth_img = camera.get_pose_img(j)

            camera.save_image(color_img, depth_img, cam_path, j)
            camera.save_pcd(color_img, loaded_depth_img, cam_path)
        
        camera_dict = camera.get_camera_dict()
        world_dict["camera"] = camera_dict

        dump_world(world_dict, world, world_save_path)

        pybullet_world.disable_real_time_simulation() # disable simulation for collision label generation

        panda = PandaArm(panda_base_pose, num_robot_config, client, area_of_interest, seed, seed_num)

        # panda.cfg_generation(ratio)
        panda.cfg_generation_invk(ratio)
        
        panda.label_generation()

        panda.save_data(world_save_path)

        panda.remove_panda()

        pybullet_world.pybullet_remove_world()

        time.sleep(1)

    
if __name__ == "__main__":
    main()
