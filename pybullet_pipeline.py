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
    
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    # Overall Parameters
    num_camera_poses = 10 # Number of camera poses per world
    num_worlds = 1 # Number of worlds
    min_obj = 8 # Minimum number of objects per world
    max_obj = 15 # Maximum number of objects per world
    model_path = "/home/chengjing/Desktop/pybullet-URDF-models/urdf_models/models"
    save_path = "/home/chengjing/Desktop/cam_test"

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
                "plastic_orange":{},
                "potato_chip_1":{},
                "scissors":{},
                "poker_1":{},
                "remote_controller_1": {},
                "glue_1": {},
                "red_marker": {},
                "soap": {},
                "small_clamp": {},
                "flat_screwdriver": {},
                "cracker_box": {},
                }

    obj_limits = {"x":(0.665, 0.965), "y":(0.2, 0.8), "z": 0.1} # Object spawning limits
    obj_z_offset = 0.63 # offset of the table

    
    
    # Camera Parameters
    camera_look_at = [0.815, 0.5, 0.63] # Camera look at point
    camera_phi =  pi *  1 / 6  # Camera phi angle
    camera_theta = 0  # Camera theta angle  BEFORE: PI
    camera_radius = 1.15# Camera radius
    camera_theta_offset = 0.25
    camera_x, camera_y, camera_z = camera_look_at[0], camera_look_at[1], camera_look_at[2]
    camera_multipler = 5 # For depth image conversion, set the lower bound of the conversion
    camera_phi_var, camera_theta_var, camera_radius_var = pi / 18, pi / 9, 0.05 # Camera phi theta and radius variance
    camera_near = 0.05 # Camera near plane
    camera_far = 1000 # Camera far plane
    # camera fov = 60, camera near 0.05, camera far 5. You can add them here and change initialization. 

    # Panda arm parameter
    area_of_interest = {"x":(0.665, 0.965), "y":(0.3, 0.6), "z": (0.63, 0.98)} # Area of interest for camera
    ratio = 0.5 # Rough ratio of in AOI and out of AOI
    panda_base_pose = [0.315, 0.5, 0.63] 
    num_robot_config = 10000
    seed = False
    seed_num = 0
    
    np.random.seed(2023)

    # temp_arm = p.loadURDF("franka_panda/panda.urdf", panda_base_pose)

    # Initialize world
    pybullet_world = PybulletWorldManager(num_worlds, object_dict, obj_limits, model_path, obj_z_offset, min_obj, max_obj)

    # Get the world list for iteration
    worlds = pybullet_world.get_world_list()
    pybullet_world.set_world_gravity()

    # Load default world with plane and table
    pybullet_world.load_default_world()
    
    cfg_initialization = True

    for i, world in enumerate(tqdm(worlds, desc = "Total World")):
        
        if client == None:  # hacky way to get opengl shadow working
            client = p.connect(p.GUI)
            pybullet_world.set_world_gravity()
            pybullet_world.load_default_world()
        
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

        camera = PybulletCamera(num_camera_poses, camera_look_at, camera_multipler, near = camera_near, far = camera_far) # Initialize camera
        camera.pose_generation(camera.num_poses, camera_theta, camera_phi, camera_radius, camera_x, camera_y, camera_z
                              ,camera_theta_var, camera_phi_var, camera_radius_var, camera_theta_offset) # Pose generation

        world_dict = {world: pybullet_world.world[world]}
        
        
        # Load panda arm         
        panda = PandaArm(panda_base_pose, num_robot_config, client, area_of_interest, seed, seed_num)
        
        # Set panda to rest pose and take picture
        panda.set_pose(panda.rest_pose)

        for j in tqdm(range(num_camera_poses), desc = f"World {i}", leave= False):
            cam_ = 'cam_' + str(j)
            cam_path = os.path.join(world_save_path, cam_) 
            os.mkdir(cam_path)

            # Get image from the jth camera pose
            color_img, depth_img, loaded_depth_img = camera.get_pose_img(j)

            camera.save_image(color_img, depth_img, cam_path, j)
            camera.save_pcd(color_img, loaded_depth_img, cam_path)
        
        camera_dict = camera.get_camera_dict()
        world_dict["camera"] = camera_dict

        dump_world(world_dict, world, world_save_path)

        pybullet_world.disable_real_time_simulation() # disable simulation for collision label generation


        # panda.cfg_generation(ratio)
        
        if cfg_initialization:
        
            panda.cfg_generation_invk_null_space_global(ratio)
            
            cfg_initialization = False
        
        else:
            # load the previous cfg file
            
            panda.load_cfgs_aoi(os.path.join(save_path, "world_0"))
        
        panda.label_generation()

        panda.save_data(world_save_path)

        panda.remove_panda()

        pybullet_world.pybullet_remove_world()

        time.sleep(1)
        
        p.disconnect()
        
        client = None

    
if __name__ == "__main__":
    main()
