import cv2
import numpy as np
from modvo.vo.tracker import Tracker
from modvo.utils.geometry import pose_from_kpts

class FrameByFrameTracker(Tracker):
    def __init__(self, **params):
        self.camera = params['camera']
        self.detector = params['detector']
        self.matcher = params['matcher']
        self.first_image = True
        self.feats0 = None
        self.feats1 = None
        self.index = 0
        self.img0, self.img1 = None, None
        self.n_matches = 0
        self.n_inliers = 0


    def track(self, image):
        if(self.index == 0):
            self.R = np.identity(3)
            self.t = np.zeros((3, 1))
            self.feats0 = self.detector.detectAndCompute(image)
            self.img0 = image
        else:
            self.img1 = image
            self.feats1 = self.detector.detectAndCompute(image)
            
            matches = self.matcher.match(self.feats0, self.feats1)
            self.n_matches = len(matches['matches'])
            print('n matches ', len(matches['matches']))
            #get matched kpts
            kpts0 = self.feats0['keypoints'][matches['matches'][:,0]] 
            kpts1 = self.feats1['keypoints'][matches['matches'][:,1]]
            mask, R, t, self.n_inliers = pose_from_kpts(kpts0, kpts1, self.camera)
            print('inliers ', self.n_inliers)

            self.t = self.t + self.R.dot(t)
            self.R = R.dot(self.R)
            
            self.feats0 = self.feats1
            self.img0 = self.img1
            
        self.index += 1

        return self.R, self.t
    
    
if __name__ == '__main__':
    from modvo.detectors import orb, superpoint
    from modvo.matchers.bf import BFMatcher
    from modvo.dataloaders.kitti import KITTILoader
    from modvo.utils.viz import *
    from modvo.gui.viewer import GUIDrawer
    from modvo.maps.kf_based import Frame

    dlparams = {'root_path': '/root/datasets/kitti/',
                'start_frame': 0,
                'stop_frame': 800,
                'sequence_name': '03',
                'camera_id': '0',}
    dataloader = KITTILoader(**dlparams)

    det_params = {'nfeatures': 1000,
                  'scaleFactor': 1.2,
                  'nlevels': 8}
    det = orb.ORBDetector(**det_params)

    matcher_params = {'normType':cv2.NORM_L1, 
                      'crossCheck': False}
    matcher = BFMatcher(**matcher_params)
    
    vo_params = {'camera': dataloader.camera,
                 'detector': det,
                 'matcher': matcher}
    vo = FrameByFrameTracker(**vo_params)
    gui = GUIDrawer()
    frames = []
    for i, img  in enumerate(dataloader):
        print(i,'/', len(dataloader))
        R, t = vo.track(img)
        f = Frame(img)
        frame_pose = np.eye(4)
        frame_pose[:3,:3] = R
        frame_pose[:3,3] = t.flatten()
        f.pose = frame_pose
        frames.append(f)
        gui.draw_map(frames=frames)