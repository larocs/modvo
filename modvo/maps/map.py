from abc import ABC, abstractmethod


class Map(ABC):
    @abstractmethod
    def __init__(self, **params):
        self.points = [None]
        self.frames = [None]
        self.point_index = 0
        self.frame_index = 0


    def get_points(self): 
        return self.points   
    

    def get_frames(self):
        return self.frames
    

    def get_num_points(self):
        return len(self.points) 


    def get_num_frames(self):
        return len(self.frames)


    @abstractmethod
    def add_point(self, point):
        '''
        Adds the points to the map
        Parameters:
           - point: point object to be added
        '''
        raise NotImplementedError


    @abstractmethod
    def add_frame(self, frame):
        '''
        Adds the frames to the map
        Parameters:
            - frame: frame to be added
        '''
        raise NotImplementedError


    @abstractmethod
    def remove_point(self, point):
        '''
        Removes the points from the map
        Parameters:
          - point: point to be removed
        '''
        raise NotImplementedError


    @abstractmethod
    def remove_frame(self, frame):
        '''
        Removes the frames from the map
        Parameters:
           - frame: frame to be removed
        '''
        raise NotImplementedError