from abc import ABC, abstractmethod

class Tracker(ABC):
    @abstractmethod
    def __init__(self, **params):
        self.camera = None
        self.detector = None
        self.matcher = None
        self.R = None
        self.t = None

    def track(self, image):
        '''
        Method to track the features and compute motion
        Args:
            image: current image
        Returns:
            Rotation matrix and translation vector
        '''
        return self.R, self.t