import cv2
import torch
import numpy as np
from modvo.detectors.detector import Detector
from modvo.thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint
torch.set_grad_enabled(False)

class SuperPointDetector(Detector):
    
    def __init__(self, **params):
        self.detector = SuperPoint(params)
        self.device = params['device']
        self.detector = self.detector.to(self.device)

    def getNLevels(self):
        return 1

    def getScaleFactor(self):
        return 1
    
    def detectAndCompute(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_tensor = torch.from_numpy(image/255.).float()[None, None].to(self.device)
        pred = self.detector({'image': image_tensor})
        self.features = {'keypoints': pred['keypoints'][0].cpu().detach().numpy(),
                         'octaves': np.zeros(pred['scores'][0].shape),
                         'descriptors': pred['descriptors'][0].cpu().detach().numpy().transpose(),
                         'scores': pred['scores'][0].cpu().detach().numpy()}
        return self.features


if __name__ == '__main__':
    from modvo.utils import viz
    img0 = cv2.imread('/root/modvo/tests/imgs/000000.png')
    params = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'path': '/root/modvo/modvo/thirdparty/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth',
        'device': 'cuda'
    }
    det = SuperPointDetector(**params)
    features = det.detectAndCompute(img0)
    kpts_img = viz.draw_keypoints(img0, features['keypoints'], color='random')
    cv2.imwrite('/root/modvo/kpts_test.png', kpts_img)