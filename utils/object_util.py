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
import warnings
warnings.filterwarnings("ignore", '.*Gimbal lock detected*.') # ignore gimbal lock warning

# NOTE: Pybullet default use quaternion for orientation, since we are using moveit for octomap collision, we collect data with euler angles.

class PybulletWorldGen():
    def __init__(self, objects, num_worlds, min_obj = 8, max_obj = 10):
        self.object_dict = objects
        objects = list(objects.keys())
        self.objects = np.array(objects) if type(objects) != np.ndarray else objects
        self.total_objects = len(self.objects)
        self.world = {}
        self.rng = default_rng()
        self.min_obj = min_obj
        self.max_obj = max_obj

        self.world_gen(num_worlds)

    def object_random_choice(self):
        if self.total_objects <= 2:
            exit("Too Few Objects!")
        num_object = np.random.randint(self.min_obj, self.max_obj+1)
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

    def __init__(self, num_worlds, tota_object_dict, object_limits, model_path, z_offset = 0.63, min_obj = 8, max_obj = 15):
        self.world_ = PybulletWorldGen(objects=tota_object_dict, num_worlds=num_worlds, min_obj=min_obj, max_obj=max_obj)
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
    def __init__(self, base_pose, num_poses, client_id, area_of_interest, seed = False, seed_num = 0, invk_threshold = 0.005):
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
        self.lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.upper_limits = [2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973]
        self.robot_range =[self.upper_limits[i] - self.lower_limits[i] for i in range(len(self.lower_limits))]
        self.rest_pose = [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]
        self.threshold = invk_threshold
    

        if seed:
            torch.manual_seed(seed_num)
        
        self.DOF = 7

        self.cfgs = torch.zeros((self.num_poses, self.DOF), dtype=torch.float32)
        self.labels = torch.zeros(self.num_poses, dtype=torch.float32)
        self.client_id = client_id      
        self.aoi = area_of_interest
        
    def load_cfgs_aoi(self, path_to_cfgs):
        self.cfgs = torch.load(os.path.join(path_to_cfgs, "robot_config.pt"))
        self.aoi_marker = torch.load(os.path.join(path_to_cfgs, "aoi_marker.pt"))
        if self.num_poses != self.cfgs.size(dim = 0):
            print("number of poses in cfgs does not match num_poses entered, overwriting num_poses")
            self.num_poses = self.cfgs.size(dim = 0)
            self.labels = torch.zeros(self.num_poses, dtype=torch.float32)
            
            
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
        
        
    def cfg_generation_invk_null_space(self, ratio = 0.5, extend_range = 0.2, max_iter = 100, residual = 0.005, reinit_iter = 5):
        self.aoi_marker = torch.zeros(self.num_poses, dtype=torch.bool)
        aoi_count, aoi_total = 0, int(self.num_poses * ratio)
        rand_count, rand_total = 0, int(self.num_poses * (1 - ratio))
        
        while aoi_count < aoi_total or rand_count < rand_total:
            xyz = self.get_coordinate(extend_range / 2)
            count = aoi_count + rand_count
            # print("aoi_count", aoi_count, "rand_count", rand_count)
            cfg = p.calculateInverseKinematics(bodyUniqueId=self.pandaID, endEffectorLinkIndex = 11,
                                             targetPosition = xyz, maxNumIterations = max_iter, residualThreshold = residual,
                                             lowerLimits = self.lower_limits, upperLimits = self.upper_limits,
                                             jointRanges = self.robot_range, restPoses = self.rest_pose)
            cfg = cfg[:7]
            
            solution_flag = False
            
            if not self.check_solution_distance(cfg, xyz, self.threshold):
                for i in range(reinit_iter):
                    if i == reinit_iter - 1:
                        self.set_pose(self.rest_pose)
                    else:
                        self.set_pose(self.get_cfg())
                    cfg = p.calculateInverseKinematics(bodyUniqueId=self.pandaID, endEffectorLinkIndex = 11,
                                             targetPosition = xyz, maxNumIterations = max_iter, residualThreshold = residual,
                                             lowerLimits = self.lower_limits, upperLimits = self.upper_limits,
                                             jointRanges = self.robot_range, restPoses = self.rest_pose)
                    cfg = cfg[:7]
                    if self.check_solution_distance(cfg, xyz, self.threshold):
                        solution_flag = True    
                        break
            else:
                solution_flag = True
            
            if not solution_flag:
                continue # reinit_iter attempt failed, skip this pose
            
            
            if not self.check_cfg(cfg):
                continue
        

            in_aoi = self.check_aoi_xyz(xyz)
            
            if in_aoi and aoi_count < aoi_total:
                aoi_count += 1
                self.aoi_marker[count] = True 
            elif not in_aoi and rand_count < rand_total:
                rand_count += 1
                self.aoi_marker[count] = False
                
            self.cfgs[count] = torch.FloatTensor(cfg)
            
    def cfg_generation_invk_null_space_global(self, ratio = 0.5, max_iter = 100, residual = 0.005, reinit_iter = 5):
        self.aoi_marker = torch.zeros(self.num_poses, dtype=torch.bool)
        aoi_count, aoi_total = 0, int(self.num_poses * ratio)
        rand_count, rand_total = 0, int(self.num_poses * (1 - ratio))
        
        while aoi_count < aoi_total or rand_count < rand_total:
            
            count = aoi_count + rand_count
            
            aoi_flag = True if torch.randn(1).item() < ratio else False
            
            if aoi_flag and aoi_count >= aoi_total:
                aoi_flag = False
            elif not aoi_flag and rand_count >= rand_total:
                aoi_flag = True
                
            if aoi_flag:
            
                xyz = self.get_coordinate(0)
                
                # print("aoi_count", aoi_count, "rand_count", rand_count)
                cfg = p.calculateInverseKinematics(bodyUniqueId=self.pandaID, endEffectorLinkIndex = 11,
                                                targetPosition = xyz, maxNumIterations = max_iter, residualThreshold = residual,
                                                lowerLimits = self.lower_limits, upperLimits = self.upper_limits,
                                                jointRanges = self.robot_range, restPoses = self.rest_pose)
                cfg = cfg[:7]
                
                solution_flag = False
                
                if not self.check_solution_distance(cfg, xyz, self.threshold):
                    for i in range(reinit_iter):
                        if i == reinit_iter - 1:
                            self.set_pose(self.rest_pose)
                        else:
                            self.set_pose(self.get_cfg())
                        cfg = p.calculateInverseKinematics(bodyUniqueId=self.pandaID, endEffectorLinkIndex = 11,
                                                targetPosition = xyz, maxNumIterations = max_iter, residualThreshold = residual,
                                                lowerLimits = self.lower_limits, upperLimits = self.upper_limits,
                                                jointRanges = self.robot_range, restPoses = self.rest_pose)
                        cfg = cfg[:7]
                        if self.check_solution_distance(cfg, xyz, self.threshold):
                            solution_flag = True    
                            break
                else:
                    solution_flag = True
                
                if not solution_flag:
                    continue # reinit_iter attempt failed, skip this pose
                
                
                if not self.check_cfg(cfg):
                    continue
                
                aoi_count += 1
                self.aoi_marker[count] = True
            
            else:
                rand_count += 1
                cfg = self.get_cfg()
                if self.check_cfg_aoi(cfg):
                    self.aoi_marker[count] = True
                else:
                    self.aoi_marker[count] = False
            
            self.cfgs[count] = torch.FloatTensor(cfg)
            
        
            
    def cfg_generation_invk(self, ratio = 0.5, extend_range = 0.2, max_iter = 100, residual = 0.005):
        # DO NOT USE THIS FUNCTION FOR NOW
        self.aoi_marker = torch.zeros(self.num_poses, dtype=torch.bool)
        aoi_count, aoi_total = 0, int(self.num_poses * ratio)
        rand_count, rand_total = 0, int(self.num_poses * (1 - ratio))
        
        while aoi_count < aoi_total or rand_count < rand_total:
            xyz = self.get_coordinate(extend_range / 2)
            count = aoi_count + rand_count
            print("aoi_count", aoi_count, "rand_count", rand_count)
            cfg = p.calculateInverseKinematics(bodyUniqueId=self.pandaID, endEffectorLinkIndex = 11,
                                             targetPosition = xyz, maxNumIterations = max_iter, residualThreshold = residual)
            cfg = cfg[:7]
            
            self.set_pose(cfg)
            
            if not self.check_cfg(cfg):
                print("not in range")
                continue
            
            in_aoi = self.check_aoi_xyz(xyz)
            
            if in_aoi and aoi_count < aoi_total:
                aoi_count += 1
                self.aoi_marker[count] = True 
            elif not in_aoi and rand_count < rand_total:
                rand_count += 1
                self.aoi_marker[count] = False
                
            self.cfgs[count] = torch.FloatTensor(cfg)
        
    def get_coordinate(self, range):
        x_low, x_high = self.aoi['x'][0] * (1 - range), self.aoi['x'][1] * (1 + range)
        y_low, y_high = self.aoi['y'][0] * (1 - range), self.aoi['y'][1] * (1 + range)
        z_low, z_high = self.aoi['z'][0] * (1 - range), self.aoi['z'][1] * (1 + range)
        
        x  = np.random.rand() * (x_high - x_low) + x_low
        y  = np.random.rand() * (y_high - y_low) + y_low
        z  = np.random.rand() * (z_high - z_low) + z_low
        
        return np.array([x, y, z])
    
    def check_cfg(self, cfg):
        for i in range(self.DOF):
            if cfg[i] < self.constraints[i][0] or cfg[i] > self.constraints[i][1]:
                return False
        return True
        
                    
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

    def check_aoi_xyz(self, xyz):
        x, y, z = xyz
        if self.aoi['x'][0] <= x <= self.aoi['x'][1] and self.aoi['y'][0] <= y <= self.aoi['y'][1] and self.aoi['z'][0] <= z <= self.aoi['z'][1]:
            return True
        return False

    def label_generation(self):
        for i in range(self.num_poses):
            self.labels[i] = set_joints_and_get_collision_status(self.pandaID, self.cfgs[i], self.client_id)
        print()
        print(f'{torch.sum(self.labels==1)} collisons, {torch.sum(self.labels==-1)} free')
        
    
    def check_solution_distance(self, cfg, xyz, threshold = 0.005):
        set_joints(self.pandaID, cfg, self.client_id)
        x, y, z = p.getLinkState(self.pandaID, 11)[0]  # Get end effector link position
        goal_x, goal_y, goal_z = xyz
        if np.sqrt((goal_x - x)**2 + (goal_y - y)**2 + (goal_z - z)**2) > threshold:
            return False
        return True

    def check_cfg_aoi(self, cfg):
        for i in range(self.DOF):
            p.resetJointState(self.pandaID, i, cfg[i])
        
        end_factor_xyz = p.getLinkState(self.pandaID, 11)[0]  # Get end effector link position
        return self.check_aoi_xyz(end_factor_xyz)
        
    def set_pose(self, cfg):
        for i in range(self.DOF):
            p.resetJointState(self.pandaID, i, cfg[i])
        
    def save_data(self, path):
        cfg_path = os.path.join(path, 'robot_config.pt')
        label_path = os.path.join(path, 'collision_label.pt')
        marker_path = os.path.join(path, 'aoi_marker.pt')
        torch.save(self.cfgs, cfg_path)
        torch.save(self.labels, label_path)
        torch.save(self.aoi_marker, marker_path)
        print("\naoi_count", torch.sum(self.aoi_marker))
        print("cfgs_verify_sum", torch.sum(self.cfgs))

    def remove_panda(self):
        p.removeBody(self.pandaID)
   