import numpy as np
from abc import ABC, abstractmethod


class Detector(ABC):
    @abstractmethod
    def __init__(self, **params):
        self.detector = None
        self.features = {'keypoints': [[None, None]],
                         'octaves': [None],
                         'descriptors': [[None]],
                         'scores': [None]}

    @abstractmethod
    def getNLevels(self):
        return NotImplementedError

    @abstractmethod
    def getScaleFactor(self):
        return NotImplementedError

    @abstractmethod
    def detectAndCompute(self, image):
        '''
        Method to detect and describe features from an image

        Args:
            image: image to detect and describe
        Returns:
            dictionary with keypoints, descriptors and scores
        '''
        
        return self.features 
    
    def getScaleLevels(self):
        '''
            Get feature scale for all levels
        '''
        scales = np.zeros(self.getNLevels())
        scales[0] = 1.0
        if(self.getNLevels() > 1):
            for i in range(1, scales.shape[0]):
                scales[i] = scales[i-1] * self.getScaleFactor()
        return scales
