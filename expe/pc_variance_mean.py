from tarfile import XGLTYPE
import numpy as np
import os
import sys
sys.path.append('../')
from utils.data_reader_util import * 

path = "/home/chengjing/Desktop/data_train"


worlds = 70
cams = 10

pc_average = []
xyz_mean = None
xyz_max = None
xyz_min = None
color_mean = None
color_max = None
color_min = None



for world in range(worlds):
        wpath = os.path.join(path, "world_" + str(world))
        for cam in range(cams):
            cpath = os.path.join(wpath, "cam_" + str(cam))
            color, pcd = get_color_pcd(cpath)
            color, pcd = np.array(color), np.array(pcd)
            pc_average.append(pcd.shape[0])
            xyz_mean = np.mean(pcd, axis = 0, keepdims=True) if np.any(xyz_mean) == None else np.vstack((xyz_mean, np.mean(pcd, axis = 0, keepdims=True)))
            xyz_max = np.max(pcd, axis = 0, keepdims=True) if np.any(xyz_max) == None else np.vstack((xyz_max, np.max(pcd, axis = 0, keepdims=True)))
            xyz_min = np.min(pcd, axis = 0, keepdims=True) if np.any(xyz_min) == None else np.vstack((xyz_min, np.min(pcd, axis = 0, keepdims=True)))
            color_mean = np.mean(color, axis = 0, keepdims=True) if np.any(color_mean) == None else np.vstack((color_mean, np.mean(color, axis = 0, keepdims=True)))
            color_max = np.max(color, axis = 0, keepdims=True) if np.any(color_max) == None else np.vstack((color_max, np.max(color, axis = 0, keepdims=True)))
            color_min = np.min(color, axis = 0, keepdims=True) if np.any(color_min) == None else np.vstack((color_min, np.min(color, axis = 0, keepdims=True)))
            
print("PC: ")
print("mean: ", np.mean(pc_average),  "\nvariance: ", np.var(pc_average), "\nmax: ", np.max(pc_average), "\nmin: ", np.min(pc_average))

print("XYZ: ")
print("mean: ", np.mean(xyz_mean, axis = 0),  "\nvariance: ", np.var(xyz_mean, axis = 0), "\nmax: ", np.max(xyz_mean, axis = 0), "\nmin: ", np.min(xyz_mean, axis = 0))

print("Color: ")
print("mean: ", np.mean(color_mean, axis = 0),  "\nvariance: ", np.var(color_mean, axis = 0), "\nmax: ", np.max(color_mean, axis = 0), "\nmin: ", np.min(color_mean, axis = 0))


"""
PC: 
mean:  207434.54428571428 
variance:  531718886.70232457 
max:  251392 
min:  168960
XYZ: 
mean:  [-0.00294618 -0.09776433  1.20235818] 
variance:  [0.00033196 0.01005051 0.03240985] 
max:  [0.03376896 0.05553655 1.50009025] 
min:  [-0.06283303 -0.29893277  0.79540847]
Color: 
mean:  [0.49552827 0.45530169 0.40686018] 
variance:  [0.00040298 0.00067269 0.00113497] 
max:  [0.54354318 0.51825189 0.48805935] 
min:  [0.43938556 0.38606734 0.32082873]
"""
            
           