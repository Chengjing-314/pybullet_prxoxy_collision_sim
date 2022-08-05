import json
import numpy as np
import h5py
from utils.general_util import * 


def get_world_dict(path):
    with open(path, "r") as f:
        world_dict = json.load(f)

    return world_dict

def get_camera_pose(world_dict, i):
    camera_dict = world_dict["camera"]
    xyz, rpy = camera_dict["pose"][i]["xyz"], camera_dict["pose"][i]["rpy"]

    return xyz, rpy

def get_camera_intrinsic(world_dict):
    camera_dict = world_dict["camera"]
    intrinsic = camera_dict["intrinsic"]

    return intrinsic

def get_camera_near_far(world_dict):
    camera_dict = world_dict["camera"]
    near, far = camera_dict["near"], camera_dict["far"]
    
    return near, far

def get_camera_multipler(world_dict):
    camera_dict = world_dict["camera"]
    multiplier = camera_dict["multipler"]

    return multiplier


def depth_image_to_true_depth(depth_image, multipler, near, far):
    uint16_min, uint16_max  = 0, 2**16 - 1
    depth_image = depth_image.astype(np.float32)
    depth_image[depth_image == 0] = np.nan
    depth_image = depth_image_range_conversion(depth_image, near * multipler, far, uint16_min, uint16_max)
    return depth_image

def get_color_pcd(path):
    pc_path = os.path.join(path, "pc.h5")
    color_path = os.path.join(path, "color.h5")
    with h5py.File(pc_path, "r") as f:
        pcd = np.array(f["pointcloud"][:])
    
    with h5py.File(color_path, "r") as f:
        color = np.array(f["color"][:])
    
    return color, pcd