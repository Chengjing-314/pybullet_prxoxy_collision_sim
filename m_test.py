import pandas
import pybullet as p
import pybullet_data as pd
import numpy as np
from utils.object_util import * 

c = p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

p.loadURDF("plane.urdf")
plateUID = p.loadURDF("data/dinnerware/plate.urdf", basePosition = [0.7, 0, 0.01])
cup1UID = p.loadURDF("data/dinnerware/cup/cup_small.urdf", basePosition = [0.85, 0.1, 0.01])
cup2UID = p.loadURDF("data/dinnerware/cup/cup_small.urdf", basePosition = [0.55, -0.1, 0.02])
duck = p.loadURDF("data/duck/duck_vhacd.urdf", basePosition = [0.5,0,0.02], baseOrientation = [0,10,0,0])



panda = PandaArm([0,0,0], 100, c)

for i in range(panda.num_poses):
        panda.labels[i] = set_joints_and_get_collision_status(panda.pandaID, panda.cfgs[i], panda.client_id)
        if panda.labels[i] == 1:
            print("In collision")
            input("Press Enter to continue...")
        else:
            continue