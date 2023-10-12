import torch
import numpy as np
from modvo.matchers.matcher import Matcher
from modvo.thirdparty.LightGlue.lightglue import LightGlue
torch.set_grad_enabled(False)

class LightglueMatcher(Matcher):  
    def __init__(self, **params):
        self.matcher = LightGlue(features=params['features']).eval()
        self.device = params['device']
        self.matcher = self.matcher.to(self.device)
        self.size_imgs = (params['image_height'], params['image_width'])

    def match(self, features0, features1):
        data0 = {}
        data0['keypoints'] = torch.Tensor(features0["keypoints"]).float().to(self.device).unsqueeze(0)
        data0['keypoint_scores'] = torch.Tensor(features0["scores"]).float().to(self.device).unsqueeze(0)
        data0['descriptors'] = torch.Tensor(features0["descriptors"]).float().to(self.device).unsqueeze(0)
        data0['image_size'] = self.size_imgs

        data1 = {}
        data1['keypoints'] = torch.Tensor(features1["keypoints"]).float().to(self.device).unsqueeze(0)
        data1['keypoint_scores'] = torch.Tensor(features1["scores"]).float().to(self.device).unsqueeze(0)
        data1['descriptors'] = torch.Tensor(features1["descriptors"]).float().to(self.device).unsqueeze(0)
        data1['image_size'] = self.size_imgs

        matches = self.matcher({'image0': data0, 'image1': data1})
        matches0 = matches['matches0'][0].cpu().numpy()
        scores0 = matches['matching_scores0'][0].cpu().detach().numpy()
        match_conf = []
        for i, (m, c) in enumerate(zip(matches0, scores0)):
            match_conf.append([i, m, c])
        match_conf = sorted(match_conf, key=lambda x: x[2], reverse=True)

        valid = [[l[0], l[1]] for l in match_conf if l[1] > -1]
        m0 = [l[0] for l in valid]
        m1 = [l[1] for l in valid]
        scores = [l[2] for l in match_conf if l[1] > -1]
        self.matches = {'matches': np.array([m0, m1]).transpose(),
                        'scores': np.array(scores)}

        return self.matches


if __name__ == '__main__':
    import cv2
    from modvo.detectors import superpoint
    from modvo.utils import viz

    img0 = cv2.imread('tests/imgs/000000.png')
    img1 = cv2.imread('tests/imgs/000001.png')

    det_params = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 400,
        'remove_borders': 4,
        'path': 'modvo/thirdparty/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth',
        'device': 'cuda'
    }

    det = superpoint.SuperPointDetector(**det_params)
    f0 = det.detectAndCompute(img0)
    f1 = det.detectAndCompute(img1)
    matcher_params = {'features': 'superpoint', 'device': 'cuda', 'image_height': img0.shape[0], 'image_width': img0.shape[1]}
    matcher = LightglueMatcher(**matcher_params)
    matches = matcher.match(f0, f1)
    
    matches_img = viz.draw_matches(img0, img1, 
                                   f0['keypoints'][matches['matches'][:,0]],
                                   f1['keypoints'][matches['matches'][:,1]], 
                                   matches['scores'])
    cv2.imwrite('modvo/match_test.png', matches_img)
