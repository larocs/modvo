import torch
import numpy as np
from modvo.matchers.matcher import Matcher
from modvo.thirdparty.SuperGluePretrainedNetwork.models.superglue import SuperGlue
torch.set_grad_enabled(False)

class SuperglueMatcher(Matcher):  
    def __init__(self, **params):
        self.matcher = SuperGlue(params)
        self.device = params['device']
        self.matcher = self.matcher.to(self.device)
        self.size_imgs = (params['image_height'], params['image_width'])
        
    def match(self, features0, features1):
        data = {}
        data['image_size'] = self.size_imgs
        
        data['scores0'] = torch.Tensor(features0["scores"]).float().to(self.device).unsqueeze(0)
        data['keypoints0'] = torch.Tensor(features0["keypoints"]).float().to(self.device).unsqueeze(0)
        data['descriptors0'] = torch.Tensor(features0["descriptors"]).float().to(self.device).unsqueeze(0).transpose(1, 2)

        data['scores1'] = torch.Tensor(features1["scores"]).float().to(self.device).unsqueeze(0)
        data['keypoints1'] = torch.Tensor(features1["keypoints"]).float().to(self.device).unsqueeze(0)
        data['descriptors1'] = torch.Tensor(features1["descriptors"]).float().to(self.device).unsqueeze(0).transpose(1, 2)
        
        matches = self.matcher(data)
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

    img0 = cv2.imread('/root/modvo/tests/imgs/000000.png')
    img1 = cv2.imread('/root/modvo/tests/imgs/000001.png')

    det_params = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 400,
        'remove_borders': 4,
        'path': '/root/modvo/modvo/thirdparty/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth',
        'device': 'cuda'
    }

    det = superpoint.SuperPointDetector(**det_params)
    f0 = det.detectAndCompute(img0)
    f1 = det.detectAndCompute(img1)
    matcher_params = {'image_size0': img0.shape, 
                      'image_size1': img1.shape,
                      'weights': 'outdoor',
                      'device': 'cuda',
                      'path': '/root/modvo/modvo/thirdparty/SuperGluePretrainedNetwork/models/weights/superglue_outdoor.pth'}
    matcher = SuperglueMatcher(**matcher_params)
    matches = matcher.match(f0, f1)

    matches_img = viz.draw_matches(img0, img1, 
                                   f0['keypoints'][matches['matches'][:,0]],
                                   f1['keypoints'][matches['matches'][:,1]], 
                                   matches['scores'])
    cv2.imwrite('/root/modvo/match_test.png', matches_img)