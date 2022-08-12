import open3d as o3d
import h5py
import os 

# Depth Not Converted

save_path = "/home/chengjing/Desktop/cam_test"
world = 1
cam = 2

world_path = os.path.join(save_path, "world_" + str(world))
cam_path = os.path.join(world_path, "cam_" + str(cam))

xyz_path = os.path.join(cam_path, "pc.h5")
color_path = os.path.join(cam_path, "color.h5")


with h5py.File(xyz_path, 'r') as f:
    xyz = f["pointcloud"][:]

with h5py.File(color_path, 'r') as f:
    color = f["color"][:]

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(color)

o3d.visualization.draw_geometries([pcd])