from abc import ABC, abstractmethod


class MapPoint(ABC):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.coordinates = [x, y, z]
        self.index = None
        self.descriptor = None

    @abstractmethod
    def update_descriptor(self, descriptor):
        raise NotImplementedError