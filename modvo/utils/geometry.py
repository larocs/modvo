import cv2
from scipy.spatial.transform import Rotation   

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
    """
    # five-point algorithm to find E
    E, mask = cv2.findEssentialMat(kpts1, kpts0, focal=camera.fx, pp=(camera.cx, camera.cy),
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
    #compute pose up to scale!
    _, R, t, _ = cv2.recoverPose(E, kpts1, kpts0, focal=camera.fx, pp=(camera.cx, camera.cy))
    return mask, R, t


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