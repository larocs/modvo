from abc import ABC, abstractmethod

class DataLoader(ABC):
    @abstractmethod
    def __init__(self, **params):
        self.root_path = None
        self.camera = None
        self.size = None
        self.index = 0

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError
               
    def __len__(self):
        return self.size
    
    @abstractmethod
    def get_timestamp(self):
        raise NotImplementedError
    
    def get_camera(self):
        return self.camera