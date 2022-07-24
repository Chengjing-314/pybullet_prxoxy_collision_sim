import pybullet as p 
import numpy as np
import pybullet_data as pd
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
import os

# NOTE: Pybullet default use quaternion for orientation, since we want to use moveit for octomap collision, we collect data with euler angles.

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
        self.spwaned_models = {}

        self.path_import()


    def path_import(self):
        for object in self._objects:
            self._objects_path[object] = os.path.joint(self._root_path, object, 'model.urdf')

    
    def spawn_model(self, object, pos, rpy):
        orn = R.from_euler('xyz', rpy).as_quat()
        model_path = self._objects_path[object]
        model_id = p.loadURDF(model_path, pos, orn)
        self.spwaned_models[object] = model_id

    def remove_model(self, object):
        model_id = self.spwaned_models[object]
        p.removeBody(model_id)