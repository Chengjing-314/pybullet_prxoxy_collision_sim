import json


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