import os
import pybullet as p
import pybullet_data as pd
from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import json


IMG_LEN = 1024
FOV = 60
NEAR = 0.1
FAR = 5.1
f = (IMG_LEN / 2) * 1 / (np.tan(np.deg2rad(FOV)/2)) # Focal length




def get_focus(Image_height, fov):
    """
    Calculate the focus of the camera.

    https://fruty.io/2019/08/29/augmented-reality-with-opencv-and-opengl-the-tricky-projection-matrix/

    Args:
        image_len (int): image length. Defaults to IMG_LEN.
        fov (int): field of view. Defaults to FOV.

    Returns:
        focus (float): focus of the camera.
    """
    return (Image_height / 2) / (np.tan(np.deg2rad(fov)/2))


def to_homog(points):
    """
    Transform points from cartesian coordinates to homogenous coordinates.

    Args:
        points (numpy ndarray): 3 * n  numpy array.

    Returns:
        homogenous transform of the coordinates, 4 * n numpy array.
    """
    N = points.shape[1]
    D = points.shape[0]
    One = np.ones((1,N))
    points_homog = np.vstack([points,One])
    return points_homog

def from_homog(points_homog):
    """
    Transform points from homogenous coordinates to cartesion coordinates.

    Args:
        points_homog (numpy ndarray): 3 * n numpy array.

    Returns:
        cartesion transform of the coordinates, 2 * n numpy array.
    """
    N = points_homog.shape[1]
    D = points_homog.shape[0]
    points_homog = points_homog / points_homog[D-1,:]
    points = np.delete(points_homog,D-1,0)
    return points

#p.computeViewMatrix(cameraEyePosition=[0, 1, 2],
      #                         cameraTargetPosition=[0, 0, 0],
     #                          cameraUpVector=[0, 0, 1])

def get_view_matrix(eye_pos, target_pos, camera_up_vec):
    """
    Calculate camera viewMatrix(extrinsic matrix)

    Args:
        eye_pos (list):  The position of the camera in world frame.
        target_pos (list): The position of the target. 
        camera_up_vec (list): The up direction of the camera.

    Returns:
       
    """
    return p.computeViewMatrix(cameraEyePosition=eye_pos,
                               cameraTargetPosition=target_pos,
                               cameraUpVector=camera_up_vec)


def get_projection_matrix(fov=FOV, aspect = 1.0, near = NEAR, far = FAR):
    return  p.computeProjectionMatrixFOV(fov=fov,
                                         aspect=aspect,
                                         nearVal=near,
                                         farVal=far)


def get_image(viewMatrix, projectionMatrix, width = IMG_LEN, height = IMG_LEN):
    """
    Get the image from the pybullet synthetic camera. 
    Args:
        viewMatrix: Camera view matrix(extrinsic matirx) from get_view_matrix.
        projectionMatrix: Camera projection matrix(OpenGL projection matrix) from get_projection_matrix.
        width: image width, Defaults to IMG_LEN.
        height: image height. Defaults to IMG_LEN.

    Returns:
        width, height, RGB_image, Depth_image, Segmentation_image
    """
    width, height, RGB_img, Depth_img, segmentation_img = p.getCameraImage(
                                                                    width=width, 
                                                                  height=height,
                                                          viewMatrix=viewMatrix,
                                              projectionMatrix=projectionMatrix)
    
    RGB_img = RGB_img[:,:,:3] # Dropped Alha Channel
    # Depth_img = Depth_img[:,:,None] # Add thrid axis

    return width, height, RGB_img, Depth_img, segmentation_img 


def get_intrin(): 
    """
    Calculate the intrinsic matrix of camera.

    Returns:
         a numpy array of the intrisinc camera
    """
    return np.array([[f, 0, (IMG_LEN-1) / 2], 
                     [0, f, (IMG_LEN-1) / 2],
                     [0, 0, 1]])
    
def get_extrin(viewMatrix):
    """
    Return the extrinsic matrix of the camera

    Args:
        viewMatrix (tuple): viewMatrix from get_view_matrix

    Returns:
        numpy array of the extrinsic matrix. 
    """
    convention_rot = Rotation.from_euler('XYZ', angles=[180, 0, 0], degrees=True).as_matrix()
    convention_tf = np.identity(4)
    convention_tf[:3, :3] = convention_rot
    corrected_extrin = convention_tf @ np.array(list(viewMatrix)).reshape((4,4)).T
    return corrected_extrin

def get_camera(extrin, width = IMG_LEN, height = IMG_LEN, f = f):
    """
    Return a camera object from extrinsic matrix from get_extrin for point cloud
    generation
    
    Args:
        extrin (numpy array): 4 x 4 extrinsic matrix from get_extrin.

    Returns:
        camera object for open3d cloud generation. 
    """
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, (width + 1)/2, (height + 1)/2)
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = extrin
    return cam

def true_z_from_depth_buffer(depthImg, far = FAR, near = NEAR):
    """
    function will take in a depth buffer from depth camera in NDC coordinate and
    convert it in to true z value. 

    https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

    Page 51
    
    Args:
        depthImg (numpy array): real depth in world frame

    """

    depthImg = far * near / (far - (far-near) * depthImg)
    
    return depthImg
    

def buffer_to_rgbd(rgbImg, depthImg):
    """
    function will convert two numpy array to o3d rgbd image for point cloud 
    generation

    Args:
        rgbImg (_type_): rgb image
        depthImg (_type_): depth image

    Returns:
        _type_: _description_
    """
    
    depth_as_img = o3d.geometry.Image((depthImg * 1000).astype(np.uint16))

    rgbd_as_img = o3d.geometry.Image((rgbImg).astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbd_as_img, 
                                                              depth_as_img, 
                                                 convert_rgb_to_intensity=False, 
                                                             depth_trunc = 1000)
    
    return rgbd


def pcd_to_mesh(pcd, downsize = True, voxel_size = 0.095):
    pcd.estimate_normals()
    if downsize:
        pcd = pcd.voxel_down_sample(voxel_size) # reduce points
    #FIXME: Borrowed Code, May Need Fix
    def round(num):
        working = str(num-int(num))
        for i, e in enumerate(working[2:]):
            if e != '0':
                return int(num) + float(working[:i+3])
    pc_mean = np.array(pcd.compute_nearest_neighbor_distance()).mean()
    scale = 0.05 * round(pc_mean)
    radii = list(np.append(np.linspace(pc_mean - 2 * scale, pc_mean - scale, 2),
            np.linspace(pc_mean, pc_mean + 4 * scale, 4)))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
    
    return mesh

def get_pcd(cam, rgbd):
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, 
                                                     cam.intrinsic, 
                                                     cam.extrinsic)

            


def place_mesh(mesh):
    saveto = './mesh.obj'
    o3d.io.write_triangle_mesh(saveto, mesh)
    visualID = p.createVisualShape(shapeType=p.GEOM_MESH, fileName="mesh.obj")
    collisionID = p.createCollisionShape(shapeType=p.GEOM_MESH,fileName="mesh.obj")
    mID = p.createMultiBody(baseCollisionShapeIndex=collisionID,
                  baseVisualShapeIndex=visualID)
    #os.remove(saveto) #FIXME: 
    return mID


def set_joints_and_get_collision_status(pandaUid, angles, clientID):
    panda_joint_id = [0,1,2,3,4,5,6]
    for i in range(len(angles)):
        p.resetJointState(pandaUid,panda_joint_id[i], angles[i])
    p.performCollisionDetection(clientID)
    c = p.getContactPoints(bodyA = pandaUid, physicsClientId = clientID)
    return 1 if c else -1 #FIXME: 



def depth_image_range_conversion(depth_image, new_min, new_max, old_min = 0, old_max = 255):
    depth_image = np.array(depth_image, dtype=np.float32)
    depth_image = (depth_image - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return depth_image


def dump_world(world_dict,world, path):
    path = os.path.join(path, world + ".json")
    with open(path, 'w') as f:
        json.dump(world_dict, f, indent = 4)



        

