from tabnanny import check
import pybullet as p 
import numpy as np
import pybullet_data as pd
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
from utils.general_util import *
import os
import time
import torch

# NOTE: Pybullet default use quaternion for orientation, since we are using moveit for octomap collision, we collect data with euler angles.

class PybulletWorldGen():
    def __init__(self, objects, num_worlds, max_objects_per_world = 8):
        self.object_dict = objects
        objects = list(objects.keys())
        self.objects = np.array(objects) if type(objects) != np.ndarray else objects
        self.total_objects = len(self.objects)
        self.world = {}
        self.rng = default_rng()
        self.max_obj = max_objects_per_world

        self.world_gen(num_worlds)

    def object_random_choice(self):
        if self.total_objects <= 2:
            exit("Too Few Objects!")
        num_object = np.random.randint(2, self.max_obj+1)
        choice = np.sort(self.rng.choice(range(self.total_objects), num_object, replace=False))
        return self.objects[choice]
    
    def world_gen(self, num_worlds):
        for i in range(num_worlds):
           world = "world_" + str(i)
           world_obj = self.object_random_choice()
           self.world[world] = {key : self.object_dict[key].copy() for key in world_obj} # shallow copy


class PybulletModelMover():
    def __init__(self, objects, limits, path):
        self.x, self.y, self.z  = limits['x'], limits['y'], limits['z']
        self.roll, self.pitch, self.yaw = (0,2 * np.pi), (0,2 * np.pi), (0,2 * np.pi)

        self._root_path = path
        self._objects = objects
        self._objects_path = {}

        self.flag = p.URDF_USE_INERTIA_FROM_FILE

        self.path_import()

    def path_import(self):
        for object in self._objects:
            self._objects_path[object] = os.path.join(self._root_path, object, 'model.urdf')

    
    def spawn_model(self, object, pos, rpy):
        orn = R.from_euler('xyz', rpy).as_quat()
        model_path = self._objects_path[object]
        model_id = p.loadURDF(model_path, pos, orn, flags=self.flag)
        return model_id

    def remove_model(self, model_id):
        p.removeBody(model_id)

    def get_model_pose(self, model_id):
        pos, orn = p.getBasePositionAndOrientation(model_id)
        rpy = R.from_quat(orn).as_euler('xyz').tolist()
        return pos, rpy

    def object_pose_generation(self, num_objects, z_offset = 0.63, num_pose = 1):
        if num_pose ==  1:
            poses = np.zeros((num_objects, 6))
        else:
            print("Not implemented yet")
        poses[:,0,None] = np.random.rand(num_objects, num_pose) * (self.x[1] - self.x[0]) + self.x[0]
        poses[:,1,None] = np.random.rand(num_objects, num_pose) * (self.y[1] - self.y[0]) + self.y[0]
        poses[:,2,None] = np.ones((num_objects, num_pose)) *  z_offset + self.z
        poses[:,3,None] = np.random.rand(num_objects, num_pose) * (self.roll[1] - self.roll[0]) + self.roll[0]
        poses[:,4,None] = np.random.rand(num_objects, num_pose) * (self.pitch[1] - self.pitch[0]) + self.pitch[0]
        poses[:,5,None] = np.random.rand(num_objects, num_pose) * (self.yaw[1] - self.yaw[0]) + self.yaw[0]

        return poses


class PybulletWorldManager():

    def __init__(self, num_worlds, tota_object_dict, object_limits, model_path, z_offset = 0.63, max_objects_per_world = 8):
        self.world_ = PybulletWorldGen(objects=tota_object_dict, num_worlds=num_worlds, max_objects_per_world=max_objects_per_world)
        self.world = self.world_.world
        self.object_limits = object_limits
        self.model_path = model_path
        self.num_worlds = num_worlds
        self.offset = z_offset
        self.spawned_models = {}

        self.world_pose_generation(z_offset=self.offset)

    def world_pose_generation(self, z_offset = 0.63):
        
        for scene in self.world.keys():
           objects = list(self.world[scene].keys())
           temp_world_mover = PybulletModelMover(objects = objects, limits = self.object_limits, path=self.model_path)
           poses = temp_world_mover.object_pose_generation(len(objects), z_offset = z_offset)
           for i, object in enumerate(objects):
            self.world[scene][object]["drop_pose"] = {"xyz": poses[i,:3].tolist(), "rpy": poses[i,3:].tolist()}


    def pybullet_set_world(self, world_name):
        objects = list(self.world[world_name].keys())
        temp_world_mover = PybulletModelMover(objects = objects,limits=self.object_limits, path=self.model_path)
        for object in objects:
            obj_pose = self.world[world_name][object]["drop_pose"]
            xyz, rpy = obj_pose["xyz"], obj_pose["rpy"]
            object_id = temp_world_mover.spawn_model(object, xyz, rpy)
            self.spawned_models[object] = object_id
            time.sleep(2)
            
        time.sleep(2)

        for object in objects:
            xyz, rpy = temp_world_mover.get_model_pose(self.spawned_models[object])
            if xyz[2] < (self.offset * 0.9): # make sure all objects are above the table 
                print("removing object: ", object)
                temp_world_mover.remove_model(self.spawned_models[object])
                self.spawned_models.pop(object)
                self.world[world_name].pop(object)
            else:
                self.world[world_name][object]["pybullet_pose"] = {"xyz": xyz, "rpy": rpy}

            

    def pybullet_remove_world(self):
        current_objects = list(self.spawned_models.keys())
        temp_world_mover = PybulletModelMover(objects = current_objects ,limits=self.object_limits, path=self.model_path)
        for object in self.spawned_models.keys():
            if object == "table":
                continue
            else:
                temp_world_mover.remove_model(self.spawned_models[object])
        self.spawned_models = {}

    def load_default_world(self):
        self.plane_id = p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", [0.75, 0.5, 0], useFixedBase=True)

    def get_world_list(self):
        return list(self.world.keys())

    def set_world_gravity(self, gravity = [0,0,-9.8]):
        p.setGravity(gravity[0], gravity[1], gravity[2])

    def enable_real_time_simulation(self):
        p.setRealTimeSimulation(1)
    
    def disable_real_time_simulation(self):
        p.setRealTimeSimulation(0)


class PandaArm():
    def __init__(self, base_pose, num_poses, client_id, area_of_interest, seed = False, seed_num = 0):
        self.base_pose = base_pose
        self.num_poses = num_poses

        self.pandaID = p.loadURDF("franka_panda/panda.urdf", base_pose, useFixedBase=True)
        self.constraints = torch.FloatTensor([[-2.8973, 2.8973],
                                             [-1.7628, 1.7628],
                                             [-2.8973, 2.8973],
                                             [-3.0718, -0.0698],
                                             [-2.8973, 2.8973],
                                             [-0.0175, 3.7525],
                                             [-2.8973, 2.8973]])

        if seed:
            torch.manual_seed(seed_num)
        
        self.DOF = 7

        self.cfgs = torch.zeros((self.num_poses, self.DOF), dtype=torch.float32)
        self.labels = torch.zeros(self.num_poses, dtype=torch.float)
        self.client_id = client_id      
        self.aoi = area_of_interest
        
    
    def cfg_generation(self, ratio = 0.5):
        self.aoi_marker = torch.zeros(self.num_poses, dtype=torch.bool)
        aoi_total, aoi_count = int(self.num_poses * ratio), 0
        rand_total, rand_count = int(self.num_poses * (1 - ratio)), 0
        while aoi_count < aoi_total:
            cfg = self.get_cfg()
            if self.check_aoi(cfg):
                aoi_count += 1
                self.cfgs[aoi_count] = cfg
            
        while rand_count < rand_total:
            cfg = self.get_cfg()
            if not self.check_aoi(cfg):
                rand_count += 1
                self.cfgs[rand_count] = cfg
                
        self.aoi_marker[:aoi_total] = True
        self.aoi_marker[aoi_total:] = False
        
        perm = torch.randperm(self.num_poses)
        
        self.cfgs = self.cfgs[perm].view(self.cfgs.size())
        self.aoi_marker = self.aoi_marker[perm].view(self.aoi_marker.size())
                    
    def get_cfg(self):
        cfg = torch.rand((self.DOF), dtype=torch.float32)
        cfg = cfg * (self.constraints[:, 1]-self.constraints[:, 0]) + self.constraints[:, 0]
        return cfg
                
    
    def check_aoi(self, cfg):
        set_joints(self.pandaID, cfg, self.client_id)
        x, y ,z = p.getLinkState(self.pandaID, 11)[0]  # Get end effector link position
        if self.aoi['x'][0] <= x <= self.aoi['x'][1] and self.aoi['y'][0] <= y <= self.aoi['y'][1] and self.aoi['z'][0] <= z <= self.aoi['z'][1]:
            return True
        return False


    def label_generation(self):
        for i in range(self.num_poses):
            self.labels[i] = set_joints_and_get_collision_status(self.pandaID, self.cfgs[i], self.client_id)
        print()
        print(f'{torch.sum(self.labels==1)} collisons, {torch.sum(self.labels==-1)} free')
        
    
    def save_data(self, path):
        cfg_path = os.path.join(path, 'robot_config.pt')
        label_path = os.path.join(path, 'collision_label.pt')
        marker_path = os.path.join(path, 'aoi_marker.pt')
        torch.save(self.cfgs, cfg_path)
        torch.save(self.labels, label_path)
        torch.save(self.aoi_marker, marker_path)

    def remove_panda(self):
        p.removeBody(self.pandaID)
   