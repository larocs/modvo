import cv2
import numpy as np
import kornia
import torch
from modvo.detectors.detector import Detector
torch.set_grad_enabled(False)


class DiskDetector(Detector):  
    def __init__(self, **params):
        self.device = params['device']
        self.detector = kornia.feature.DISK.from_pretrained(params['weights'], device=self.device)
        self.max_keypoints = params['max_keypoints'] if 'max_keypoints' in params else None
        self.window_size = params['window_size'] if 'window_size' in params else 5
        self.score_threshold = params['score_threshold'] if 'score_threshold' in params else 0.0
        self.pad_if_not_divisible = params['pad_if_not_divisible'] if 'pad_if_not_divisible' in params else False
        
    def getNLevels(self):
        return 1

    def getScaleFactor(self):
        return 1
    
    def detectAndCompute(self, image):
        #convert to channels first
        if(image.shape[2] == 3):
            image = np.transpose(image, (2,0,1))
        elif(image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        image_tensor = torch.from_numpy(image/255.).float()[None].to(self.device)
        with torch.no_grad():
            pred = self.detector(image_tensor, 
                                n=self.max_keypoints, 
                                window_size=self.window_size, 
                                score_threshold=self.score_threshold, 
                                pad_if_not_divisible=self.pad_if_not_divisible)
        
        kpts = torch.stack([p.keypoints for p in pred], 0).squeeze(0)
        scores = torch.stack([p.detection_scores for p in pred],0).squeeze(0)
        descs = torch.stack([p.descriptors for p in pred],0).squeeze(0)
        octaves = torch.zeros(scores.shape)
       
        self.features = {'keypoints': kpts.contiguous().cpu().detach().numpy(),
                         'octaves': octaves.contiguous().cpu().detach().numpy(),
                         'descriptors': descs.contiguous().cpu().detach().numpy(),
                         'scores': scores.contiguous().cpu().detach().numpy()}

        return self.features


if __name__ == '__main__':
    from modvo.utils import viz
    import kornia
    img0 = cv2.imread('tests/imgs/000000.png')
  
    params = {'weights': 'depth', 'max_keypoints': 2048, 'device': 'cuda', 'pad_if_not_divisible': True}
    det = DiskDetector(**params)
    features = det.detectAndCompute(img0)
    print('kpts ', features['keypoints'].shape)
    print('descs ', features['descriptors'].shape)
    kpts_img = viz.draw_keypoints(img0, features['keypoints'], color='random')
    cv2.imwrite('modvo/kpts_test.png', kpts_img)
