import cv2
import numpy as np
from modvo.maps.map import Map
from modvo.maps.frame import Frame as F
from modvo.maps.map_point import MapPoint as MP


class Point(MP):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.descriptor_hist = []
        self.descriptor = None
        

    def update_descriptor(self, descriptor, norm_type):
        '''
        Updates the descriptor of this map point based on the previous descriptors
            - descriptor: the descriptor array
            - norm_type: the norm type that should be used by opencv to compute the distance
        '''
        self.descriptor_hist.append(descriptor)

        if(len(self.descriptor_hist) < 2):
            self.descriptor = descriptor
        else:
            median_distances = [np.median(cv2.norm(desc, descriptor, norm_type)) for desc in self.descriptor_hist]
            self.descriptor = self.descriptor_hist[np.argmin(median_distances)].copy()


    def get_descriptor(self):
        return self.descriptor


class Frame(F):
    def __init__(self, image):
        super().__init__(image)
        self.map_points = []

    def set_pose(self, pose):
        self.pose = pose
        self.R = self.pose[:3,:3]
        self.t = self.pose[:3,3]


    def set_camera(self, camera):
        self.camera = camera


    def set_pose_from_Rt(self, R, t):
        '''
        Sets the pose of the frame in the map given matrix R and vector t
        Parameters:
           - R: 3x3 Rotation matrix
           - t: 3x1 translation vector
        '''
        self.pose[:3,:3] = R
        self.pose[:3, 3] = t.T
        self.R = R
        self.t = t

    def set_map_points_and_descs(self, points, descs, norm_type=cv2.NORM_L2):
        '''
        Sets the map points visible by this frame and update its descriptors
        Parameters:
           - points: the map points visible by this frames
           - descs: the descriptors of the points computed in this frame
           - norm_type: the norm type used by opencv to compute the distance between descriptors
        '''
        self.map_points = points
        if(self.is_keyframe):
            for p, d in zip(self.map_points, descs):
                p.update_descriptor(d, norm_type)

    
    def set_features(self, features):
        assert(isinstance(features, dict))
        self.features = features
        self.keypoints = self.features['keypoints']
    
    @property
    def keypoints(self):
        return self._kpts

    @property
    def keypoints_u(self):
        return self._kpts_u
    
    @property
    def keypoints_un(self):
        return self._kpts_un
    
    @keypoints.setter
    def keypoints(self, kpts):
        self._kpts = kpts
        if(self.camera is not None):
            self._kpts_u = self.camera.undistort_keypoints(self._kpts)
            self._kpts_un = self.camera.normalize_keypoints(self._kpts_u)

    def set_map_points(self, points):
        self.map_points = points


    def remove_features(self, mask):
        assert(self.features is not None)
        for k in self.features:
            self.features[k] = self.features[k][~mask]


    def get_num_features(self):
        assert(self.features is not None and 'keypoints' in self.features)
        return len(self.features['keypoints'])


    def project_points_to_frame(self, points3D):
        
        '''
        Project 3D points onto a 2D image
        Parameters:
            - points3D: a Nx3 array of 3D points
        Returns:
            - mask: Array of N elements, every element of which is set to 0 
                    for negative depths and to 1 for the other points
            - points_f: a Mx2 array of 2D points in frame coordinates
        '''
        Rcw, tcw = self.pose[:3,:3], self.pose[:3,3]
        # Transform the 3D points to the camera coordinate system
        points3D = Rcw @ points3D.T + tcw.reshape((3,1))
        # Project the 3D points onto the image plane
        points_f = self.camera.K @ points3D
        # Normalize the projected 2D points by the Z coordinate
        zs = points_f[-1]
        points_f = points_f[:2]/zs
        mask = zs > 0
        return mask, points_f.T


class KFBasedMap(Map):
    def __init__(self, **params):
        self.points = []
        self.frames = []
        self.keyframes = []
        self.point_index = 0
        self.frame_index = 0
        self.keyframe_index = 0
    

    def add_keyframe(self, keyframe):
        assert(isinstance(keyframe, Frame))
        self.keyframe_index = self.keyframe_index + 1
        keyframe.index = self.keyframe_index
        self.keyframes.append(keyframe)


    def remove_keyframe(self, keyframe):
        assert(isinstance(keyframe, Frame))
        self.keyframes.remove(keyframe)


    def get_num_keyframes(self):
        return len(self.keyframes)


    def get_keyframes(self):
        return self.keyframes
    

    def add_point(self, point):
        assert(isinstance(point, Point))
        self.point_index = self.point_index + 1
        point.index = self.point_index
        self.points.append(point)


    def add_points_from_coordinates(self, points):
        '''
        Adds an array of points to the map
        Parameters:
           - points: array of points coordinates
        '''
        for p in points:
            point = Point(p[0], p[1], p[2])
            self.add_point(point)


    def add_frame(self, frame):
        assert(isinstance(frame, F))
        self.frame_index = self.frame_index + 1
        frame.index = self.frame_index
        self.frames.append(frame)


    def remove_point(self, point):
        assert(isinstance(point, Point))
        self.points.remove(point)


    def remove_frame(self, frame):
        assert(isinstance(frame, F))
        self.frames.remove(frame)


    def reset(self):
        self.points = []
        self.frames = []
        self.keyframes = []
        self.point_index = 0
        self.frame_index = 0
        self.keyframe_index = 0


    def find_bad_points(self):
        pass