import cv2
import numpy as np
from collections import defaultdict
from modvo.matchers.matcher import Matcher

class BFMatcher(Matcher):  
    def __init__(self, **params):
        self.matcher = cv2.BFMatcher(**params)

    def match(self, features0, features1, ratio=None):
        matches = self.matcher.knnMatch(features0['descriptors'],
                                        features1['descriptors'],
                                        k=2)
        
        good_matches = []
        dist_match = defaultdict(lambda: float('inf'))
        index_match = {}
        for (m1, m2) in matches:
            if ratio is not None:
                if m1.distance > ratio * m2.distance:
                    continue

                if m1.trainIdx in index_match:
                    if m1.distance < dist_match[m1.trainIdx]:
                        index = index_match[m1.trainIdx]
                        good_matches[index] = m1
                        dist_match[m1.trainIdx] = m1.distance
                else:
                    dist_match[m1.trainIdx] = m1.distance
                    good_matches.append(m1)
                    index_match[m1.trainIdx] = len(good_matches) - 1
            else:
                good_matches.append(m1)

        matches = sorted(good_matches, key=lambda x: x.distance)
        matches_arr = np.zeros([len(matches), 2], dtype=np.uint)
        match_dist = np.zeros([len(matches)], dtype=float)
        
        for i, m in enumerate(matches):
            matches_arr[i, :] = [int(m.queryIdx), int(m.trainIdx)]
            match_dist[i] = m.distance

        self.matches = {
            'matches': matches_arr,
            'scores': match_dist
        }

        return self.matches


if __name__ == '__main__':
    from modvo.detectors import orb
    from modvo.utils import viz

    img0 = cv2.imread('tests/imgs/000000.png')
    img1 = cv2.imread('tests/imgs/000001.png')

    det_params = {'nfeatures': 1000,
                  'scaleFactor': 1.2,}

    det = orb.ORBDetector(**det_params)
    f0 = det.detectAndCompute(img0)
    f1 = det.detectAndCompute(img1)
    matcher_params = {'normType':cv2.NORM_L1, 
                      'crossCheck': False}
    matcher = BFMatcher(**matcher_params)
    matches = matcher.match(f0, f1)
    
    matches_img = viz.draw_matches(img0, img1, 
                                   f0['keypoints'][matches['matches'][:,0]],
                                   f1['keypoints'][matches['matches'][:,1]], 
                                   None)
    cv2.imwrite('/root/modvo/match_test.png', matches_img)