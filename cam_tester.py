from pydoc import plainpager
import pybullet as p 
import numpy as np
import pybullet_data as pd
from numpy.random import default_rng
from scipy.spatial.transform import Rotation as R
import os
import time
from PIL import Image


from utils.camera_util import PybulletCamera
from utils.general_util import get_image, get_projection_matrix, true_z_from_depth_buffer

p.connect(p.GUI)


p.setAdditionalSearchPath(pd.getDataPath())


plane_id = p.loadURDF("plane.urdf")
# table = p.loadURDF("table/table.urdf", [0.75, 0.5, 0], useFixedBase=True)


# camera = PybulletCamera(5, [1,0,0])


# path = "/home/chengjing/Desktop/img_save_test"


# while 1:
#     p.stepSimulation()
#     for i in range(camera.poses.shape[0]):
#         color_img, depth_img = camera.get_pose_img(i)
#         color_img = Image.fromarray(color_img)

#         print(depth_img)
    
#         # depth_img = Image.fromarray(depth_img)
#         # color_img.save(os.path.join(path, "color_%d.png" % i))
#         # depth_img.save(os.path.join(path, "depth_%d.png" % i))

#     time.sleep(1)

p.setRealTimeSimulation(1)


view_matrix = p.computeViewMatrix([0,0,2],[0,0,0], [0,1,0])
pm = get_projection_matrix(60)

f = np.array(pm).reshape(4,4).T[0][0]

print(f, true_z_from_depth_buffer(f))


_,_, c, d, _ = get_image(view_matrix, pm)

time.sleep(2)

print(d)

d = true_z_from_depth_buffer(d)

print(d)

print(np.min(d), np.max(d), np.mean(d))

import matplotlib.pyplot as plt

while True:
   plt.imshow(c)
   plt.show()