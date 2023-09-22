from abc import ABC, abstractmethod


class Matcher(ABC):
    @abstractmethod
    def __init__(self, **params):
        self.matcher = None
        self.matches = {'matches': [[None, None]], 
                        'scores': [None]}

    @abstractmethod
    def match(self, features0, features1):
        '''
        Method to match a pair of features extracted with the Detector

        Args:
            features0: dict with features from the first image
            feature1: dict with features form the second image
        Returns:
            dict with matches and scores
        '''
        
        return self.matches 