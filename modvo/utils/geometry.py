import cv2
import numpy as np
from scipy.spatial.transform import Rotation   
from modvslam.utils.tools import get_index

def pose_from_kpts(kpts0, kpts1, camera):
    """
    Estimates up to scale camera pose from 2D point correspondences
    Parameters:
        - kpts0: Keypoints for the first image
        - kpts1: Keypoints for the second image
        - camera: Camera object
    Returns:
        - mask:	Output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points
        - R: Output rotation matrix
        - t: Output translation vector
        - n_inliers: Number of inliers
    """
    # five-point algorithm to find E
    E, mask = cv2.findEssentialMat(kpts1, kpts0, focal=camera.fx, pp=(camera.cx, camera.cy),
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
    #compute pose up to scale!
    n_inliers, R, t, _ = cv2.recoverPose(E, kpts1, kpts0, focal=camera.fx, pp=(camera.cx, camera.cy))
    return mask, R, t, n_inliers


def triangulate_points(kpts0, kpts1, P1, P2):
    '''
    Triangulate 3D points from 2D point correspondences
    Parameters:
        - kpts0: 2xN matrix of points for the first image
        - kpts1: 2xN matrix of points for the second image
        - P1: 3x4 projection camera matrix for the first image
        - P2: 3x4 projection camera matrix for the second image
    Returns:
        - points3D: Nx3 array of N 3D points
    '''
    # Triangulate the points
    points3D_h = cv2.triangulatePoints(P1, P2, kpts0, kpts1)
    # Convert homogeneous coordinates to cartesian
    points3D = points3D_h[:3, :] / points3D_h[3, :]

    return points3D.T


def match_3D_to_2D(frame, map_points, max_reproj_distance, desc_norm_type, max_descriptor_distance):
    '''
        Matches 3D points to 2D points in the current frame
    '''
    from modvo.utils.viz import draw_keypoints

    #project 3D points into current frame
    points3D = np.array([p.coordinates for p in map_points])
    mask1, projected_pts = frame.project_points_to_frame(points3D)
    mask2, _ =  frame.camera.get_visible_pts(projected_pts)
    
    #get feature scale for all levels
    scales = frame.detector.getScaleLevels()
    
    radiuses = max_reproj_distance * scales
    proj_ids = []
    feat_ids = []
    for i, pt in enumerate(projected_pts):
        if(mask1[i] == False or mask2[i] == False):
            continue
        # Search for matching keypoints within the given radius
        indices = (abs(frame.features['keypoints'][:, 0] - pt[0]) < radiuses[frame.features['octaves']]) & \
                    (abs(frame.features['keypoints'][:, 1] - pt[1]) < radiuses[frame.features['octaves']])
        if(sum(indices) == 0):
            continue
        # Compute the distances between the projected point and the descriptors of the matching keypoints
        distances = [cv2.norm(desc, map_points[i].get_descriptor(), desc_norm_type) 
                        for desc in frame.features['descriptors'][indices]]

        # Find the closest matching keypoint
        min_dist = np.min(np.array(distances))
        min_dist_idx = np.argmin(np.array(distances))
        match_idx = get_index(frame.features['descriptors'], frame.features['descriptors'][indices][min_dist_idx])
        if(match_idx != None and match_idx not in feat_ids):
           if(min_dist < max_descriptor_distance):
               proj_ids.append(i)
               feat_ids.append(match_idx)
    return proj_ids, feat_ids


def matrix_to_quaternion(R):
    '''
    Convert rotation matrix to quaternion
    Parameters:
        - R: 3x3 rotation matrix
    Returns:
        - q: 4x1 quaternion
    '''
    r = Rotation.from_matrix(R)
    q = r.as_quat()
    return q
