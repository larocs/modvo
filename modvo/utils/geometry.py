import cv2
   

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
    E, mask = cv2.findEssentialMat(kpts1, kpts0, focal=1, pp=(0, 0),
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)       
    #compute pose up to scale!
    _, R, t, _ = cv2.recoverPose(E, kpts1, kpts0, focal=1, pp=(0, 0))
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
