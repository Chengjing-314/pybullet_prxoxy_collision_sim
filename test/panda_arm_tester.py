import pybullet as p 
import pybullet_data as pd

p.connect(p.GUI)


p.setAdditionalSearchPath(pd.getDataPath())


plane_id = p.loadURDF("plane.urdf")
panda_arm = p.loadURDF("franka_panda/panda.urdf", useFixedBase = True, basePosition = [1,1,10])

reset_pose = [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397]

for i in range(7):
    p.resetJointState(panda_arm, i, reset_pose[i])

for i in range(p.getNumJoints(panda_arm)):
    print(p.getJointInfo(panda_arm, i))
    print("\n")
    print(p.getLinkState(panda_arm, i)[4])
    print(p.getLinkState(panda_arm, i)[0])
    

input("Press Enter to continue...")