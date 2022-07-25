import pybullet as p 
import numpy as np
import pybullet_data as pd
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
import os
import time

# NOTE: Pybullet default use quaternion for orientation, since we are using moveit for octomap collision, we collect data with euler angles.

class PybulletWorldGen():
    def __init__(self, objects, num_worlds):
        self.object_dict = objects
        objects = list(objects.keys())
        self.objects = np.array(objects) if type(objects) != np.ndarray else objects
        self.total_objects = len(self.objects)
        self.world = {}
        self.rng = default_rng()

        self.world_gen(num_worlds)

    def object_random_choice(self):
        if self.total_objects <= 2:
            exit("Too Few Objects!")
        num_object = np.random.randint(2,self.total_objects + 1)
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

        self.path_import()

    def path_import(self):
        for object in self._objects:
            self._objects_path[object] = os.path.join(self._root_path, object, 'model.urdf')

    
    def spawn_model(self, object, pos, rpy):
        orn = R.from_euler('xyz', rpy).as_quat()
        model_path = self._objects_path[object]
        model_id = p.loadURDF(model_path, pos, orn)
        return model_id

    def remove_model(self, model_id):
        p.removeBody(model_id)

    def get_model_pose(self, model_id):
        pos, orn = p.getBasePositionAndOrientation(model_id)
        rpy = R.from_quat(orn).as_euler('xyz').tolist()
        return pos, rpy

    def object_pose_generation(self, num_objects, z_offset = 1.2, num_pose = 1):
        if num_pose ==  1:
            poses = np.zeros((num_objects, 6))
        else:
            print("Not implemented yet")
        poses[:,0,None] = np.random.rand(num_objects, num_pose) * (self.x[1] - self.x[0]) + self.x[0]
        poses[:,1,None] = np.random.rand(num_objects, num_pose) * (self.y[1] - self.y[0]) + self.y[0]
        poses[:,2,None] = np.ones((num_objects, num_pose)) *  z_offset
        poses[:,3,None] = np.random.rand(num_objects, num_pose) * (self.roll[1] - self.roll[0]) + self.roll[0]
        poses[:,4,None] = np.random.rand(num_objects, num_pose) * (self.pitch[1] - self.pitch[0]) + self.pitch[0]
        poses[:,5,None] = np.random.rand(num_objects, num_pose) * (self.yaw[1] - self.yaw[0]) + self.yaw[0]

        return poses


class PybulletWorldManager():

    def __init__(self, num_worlds, tota_object_dict, object_limits, model_path = '/home/chengjing/gazebo_models/', z_offset = 1):
        self.world_ = PybulletWorldGen(objects=tota_object_dict, num_worlds=num_worlds)
        self.world = self.world_.world
        self.object_limits = object_limits
        self.model_path = model_path
        self.num_worlds = num_worlds
        self.offset = z_offset
        self.spawned_models = {}

        self.world_pose_generation()

    def world_pose_generation(self):
        
        for scene in self.world.keys():
           objects = list(self.world[scene].keys())
           temp_world_mover = PybulletModelMover(objects = objects, limits = self.object_limits, path=self.model_path)
           poses = temp_world_mover.object_pose_generation(len(objects))
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
            time.sleep(0.5)
            

        for object in objects:
            xyz, rpy = temp_world_mover.get_model_pose(self.spawned_models[object])
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
    
