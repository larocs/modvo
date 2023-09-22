from abc import ABC, abstractmethod
import numpy as np

class Frame(ABC):
    @abstractmethod
    def __init__(self, image):
        self.image = image
        self.index = None
        self.camera = None
        self.pose = np.identity(4) #4x4 matrix
        self.R = self.pose[:3,:3]
        self.t = self.pose[:3,3]
        self.features = None
        self.detector = None
        self.map_points = [None]
        self.is_keyframe = False
    
    @abstractmethod
    def set_camera(self, camera):
        '''
        Sets the camera object from this frame
        Parameters:
           - camera: the camera object
        '''
        raise NotImplementedError
    
    @abstractmethod
    def set_pose(self, pose):
        '''
        Sets the pose of the frame in the map
        Parameters:
           - pose: pose of the frame
        '''
        raise NotImplementedError

    @abstractmethod
    def set_features(self, features):
        '''
        Sets the features computed in this frame
        Parameters:
           - features: features computed in this frame
        '''
        raise NotImplementedError

    @abstractmethod
    def set_map_points(self, points):
        '''
        Sets the map points visible by this frame
        Parameters:
           - points: points visible by this frame
        '''
        raise NotImplementedError
    
    def get_num_map_points(self):
        return len(self.map_points)


