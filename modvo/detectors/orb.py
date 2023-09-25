import cv2
import numpy as np
from modvo.detectors.detector import Detector
from modvo.utils import tools


class ORBDetector(Detector):  
    def __init__(self, **params):
        self.detector = cv2.ORB_create(**params)

    def getNLevels(self):
        return self.detector.getNLevels()

    def getScaleFactor(self):
        return self.detector.getScaleFactor()
    
    def detectAndCompute(self, image):
        [kpts_cv, descriptors] = self.detector.detectAndCompute(image, None)
        keypoints = tools.convert_kpts_cv_to_numpy(kpts_cv)
        octaves = np.array([kpt.octave for kpt in kpts_cv])
        scores = np.array([kpt.response for kpt in kpts_cv])

        self.features = {'keypoints': keypoints,
                         'octaves'   : octaves,
                         'descriptors': descriptors,
                         'scores': scores}
        return self.features


if __name__ == '__main__':
    from modvo.utils import viz

    img0 = cv2.imread('tests/imgs/000000.png')
    params = {'nfeatures': 1000,
              'scaleFactor': 1.2,}
    det = ORBDetector(**params)
    features = det.detectAndCompute(img0)
    kpts_img = viz.draw_keypoints(img0, features['keypoints'], color='random')
    #io.imshow(kpts_img)
    #io.show()