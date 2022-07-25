import pybullet as p 
import pybullet_data as pd

p.connect(p.GUI)


p.setAdditionalSearchPath(pd.getDataPath())


plane_id = p.loadURDF("plane.urdf")
panda_arm = p.loadURDF("franka_panda/panda.urdf", useFixedBase = True)

for i in range(p.getNumJoints(panda_arm)):
    print(p.getJointInfo(panda_arm, i))
