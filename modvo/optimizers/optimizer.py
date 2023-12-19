from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def __init__(self, **params):
        pass

    @abstractmethod
    def optimize(self, frame):
        '''
        Method to optimize the pose of a frame

        Args:
            frame: Frame object
        Returns:
            int: number of inliers
        '''
        pass