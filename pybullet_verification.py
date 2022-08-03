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


state = p.getLinkState(arm, 11)
js = p.getJointState(arm, 11)
print(state[0], js[0])


# input("Enter to continue")