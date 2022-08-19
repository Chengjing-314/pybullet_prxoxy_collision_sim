import open3d as o3d
import h5py

with h5py.File("/home/chengjing/Desktop/cam_test/world_0/cam_2/pc.h5", 'r') as f:
    xyz = f["pointcloud"][:]

with h5py.File("/home/chengjing/Desktop/cam_test/world_0/cam_2/color.h5", 'r') as f:
    color = f["color"][:]

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(color)


o3d.visualization.draw_geometries([pcd])