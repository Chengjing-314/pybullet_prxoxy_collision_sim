import os
import pybullet as p
import pybullet_data as pd 
import torch 
from  utils.general_util import * 

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

p.loadURDF('plane.urdf')
arm = p.loadURDF('franka_panda/panda.urdf', [0.315, 0.5, 0.63], useFixedBase=True)



data_path = '/home/chengjing/Desktop/img_save_test'
world_path = os.path.join(data_path, 'world_0')
robot_config = torch.load(os.path.join(world_path, 'robot_config.pt'))
collision_label = torch.load(os.path.join(world_path, 'collision_label.pt'))


cfg = robot_config[2]


set_joints_and_get_collision_status(arm, cfg, client)

time.sleep(2)

print(cfg)

state = p.getLinkState(arm, 11)
# js = p.getJointState(arm, 11)
print(state[0])

set_joints_and_get_collision_status(arm, robot_config[5], client)

#(0.45327315603738405, -0.2859920649385343, 0.6821275331717283)

m = p.calculateInverseKinematics(arm, 11, targetPosition=[0.45327315603738405, -0.2859920649385343, 0.6821275331717283], 
                                 residualThreshold=0.01, maxNumIterations = 100)

cfgg = m[:7]

# print(cfgg)

set_joints_and_get_collision_status(arm, cfgg, client)

time.sleep(2)

state = p.getLinkState(arm, 11)

print(state[0])

cfg = robot_config[3].numpy().tolist()

cfg.append([0.0, 0.0])

m = p.calculateInverseKinematics(arm, 11, targetPosition=[0.45327315603738405, -0.2859920649385343, 0.6821275331717283], 
                                 residualThreshold=0.01, maxNumIterations = 100)

# print(m)

cfgg = m[:7]

# print(cfgg)

set_joints_and_get_collision_status(arm, cfgg, client)

state = p.getLinkState(arm, 11)

print(state[0])


set_joints_and_get_collision_status(arm, robot_config[6], client)

cur_cfg = robot_config[12].numpy().tolist()
cur_cfg.append([0.0, 0.0])

# m = p.calculateInverseKinematics(arm, 11, targetPosition=[0.45327315603738405, -0.2859920649385343, 0.6821275331717283], 
#                                 currentPOsition = cur_cfg,residualThreshold=0.01, maxNumIterations = 100)

print(p.calculateInverseKinematics.func_code.co_varnames)


fgg = m[:7]


set_joints_and_get_collision_status(arm, cfgg, client)

state = p.getLinkState(arm, 11)

print(state[0])



# print(m)
# print(cfg)

# print(len(m), m)


input("Enter to continue")