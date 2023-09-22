from abc import ABC, abstractmethod

class Camera(ABC):
    @abstractmethod
    def __init__(self, **params):
        self.width = None
        self.height = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.D = [None, None, None, None, None] #k1 k2 p1 p2 k3
        self.K = [[None, None, None],
                  [None, None, None], 
                  [None, None, None]] 
        self.is_distorted = None 