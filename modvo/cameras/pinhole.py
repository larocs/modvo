import cv2
import numpy as np
from modvo.cameras.camera import Camera

class PinholeCamera(Camera):
    def __init__(self, **params):
        self.width = params['width']
        self.height = params['height']
        self.fx = params['fx']
        self.fy = params['fy']
        self.cx = params['cx']
        self.cy = params['cy']

        self.D = np.array([params['k1'], params['k2'], params['p1'], params['p2'], params['k3']])
        self.K = np.array([[self.fx, 0, self.cx],
                           [0, self.fy, self.cy],
                           [0, 0, 1]])
        self.is_distorted = np.linalg.norm(self.D) > 1e-10
    

    def undistort_keypoints(self, kpts):
        """
        Undistorts keypoints using the camera matrix and distortion coefficients
        Parameters:
            - keypoints: a Nx2 array of keypoints in the image
        Returns:
            - undistorted_keypoints: a Nx2 array of undistorted keypoints
        """
        if self.is_distorted:
            h, w = kpts.shape
            # Prepare an array to hold the undistorted keypoints
            undistorted = np.zeros((h, 2), dtype=np.float32)
            # Undistort the keypoints
            cv2.undistortPoints(kpts, self.K, self.D, undistorted)
            return undistorted
        return kpts


    def normalize_keypoints(self, points):
        """
        Normalize a set of keypoints
        Parameters:
            - points: a Nx2 array of points
        Returns:
            - normalized_keypoints: a Nx2 array of normalized points
        """
        normalized_points = np.zeros_like(points)
        normalized_points[:, 0] = (points[:, 0] - self.cx) / self.fx
        normalized_points[:, 1] = (points[:, 1] - self.cy) / self.fy
        return normalized_points


    def get_visible_pts(self, pts):
        """
        Get the points that are visible in a given camera
        Parameters:
            - pts: a Nx2 array of points
        Returns:
            - visible: Array of N elements, every element of which is set to 0 for 
                        non-visible and to 1 for visible points
            - visible_keypoints: a Nx2 array of visible points
        """

        visible = (pts[:, 0] > 0) & (pts[:, 0] < self.width) & \
                    (pts[:, 1] > 0) & (pts[:, 1] < self.height)
        return visible, pts[visible]
