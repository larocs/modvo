import cv2
import numpy as np
from modvo.vo.tracker import Tracker
from modvo.utils.geometry import pose_from_kpts, triangulate_points, match_3D_to_2D
from modvo.maps.kf_based import Frame, KFBasedMap
from modvo.optimizers.g2o import G2OOptimizer

from modvo.utils.viz import draw_keypoints

class VOLocalOptimization(Tracker):
    def __init__(self, **params):
        self.camera = params['camera']
        self.detector = params['detector']
        self.matcher = params['matcher']
        self.desc_norm_type = params['desc_norm_type']
        self.max_reproj_distance = params['max_reproj_distance']
        self.min_matches_projection = params['min_matches_projection']
        self.max_descriptor_distance = params['max_descriptor_distance']
        self.frame0 = None
        self.frame1 = None
        self.index = 0
        self.img0, self.img1 = None, None
        self.n_matches = 0
        self.n_inliers = 0
        self.window_size = 3
        self.map = KFBasedMap()
        self.optimizer = G2OOptimizer()
        self.all_pts3D = []
        self.vel = np.eye(4)
        
        

    def track(self, image):
        frame = Frame(image)
        frame.set_camera(self.camera)
        frame.detector = self.detector
        feats = self.detector.detectAndCompute(image)
        frame.set_features(feats)
        self.map.add_frame(frame)
        frame.is_keyframe = True #all frames are keyframes in this approach
        self.frame1 = frame

        if(self.index == 0):
            #First frame
            self.R = np.identity(3)
            self.t = np.zeros((3, 1))
            self.frame0 = frame
        elif(self.index == 1):
            self.feats0 = self.detector.detectAndCompute(self.frame0.image)
            self.feats1 = self.detector.detectAndCompute(self.frame1.image)
            matches = self.matcher.match(self.feats0, self.feats1)
            print('n matches ', len(matches['matches']))
            
            #get matched kpts
            kpts0 = self.feats0['keypoints'][matches['matches'][:,0]] 
            kpts1 = self.feats1['keypoints'][matches['matches'][:,1]]
            self.frame0.set_features(self.feats0)
            self.frame1.set_features(self.feats1)
            
            mask, R, t, self.n_inliers = pose_from_kpts(kpts0, kpts1, self.camera)
            self.t = self.t + self.R.dot(t)
            self.R = R.dot(self.R)
            
            self.frame1.set_pose_from_Rt(R.T, np.matmul(-R.T, t))

            inliers_mask = (mask.ravel() == 1)
            print('inliers ', self.n_inliers)

            #undistort and normalize keypoints before triangulation
            in_kpts0u = self.frame0.camera.undistort_keypoints(kpts0[inliers_mask].T)
            in_kpts1u = self.frame1.camera.undistort_keypoints(kpts1[inliers_mask].T)
            in_kpts0un = self.frame0.camera.normalize_keypoints(in_kpts0u.T)
            in_kpts1un = self.frame1.camera.normalize_keypoints(in_kpts1u.T)
            
            points3D = triangulate_points(in_kpts0un.T, in_kpts1un.T, self.frame0.pose[:3,:], self.frame1.pose[:3,:])
            print('points3D ', points3D.shape)
            #filter points based on depth consistency
            mask0, pts2d_0 = self.frame0.project_points_to_frame(points3D)
            mask1, pts2d_1 = self.frame1.project_points_to_frame(points3D)
            valid_ids = (mask0 & mask1)
            points3D = points3D[valid_ids]
            self.map.add_points_from_coordinates(points3D)
            #points3D_world = self.frame0.pose[:3,:].dot(np.vstack((points3D.T, np.ones((1, points3D.shape[0])))))
            #self.map.add_points_from_coordinates(points3D_world.T)
            descs1_valid = self.feats1['descriptors'][matches['matches'][inliers_mask,1]][valid_ids]
            self.frame1.set_map_points_and_descs(self.map.get_points(), descs1_valid)
            print('frame 1 map points ', len(self.frame1.map_points))
        else:
            #compute features
            self.feats0 = self.detector.detectAndCompute(self.frame0.image)
            self.feats1 = self.detector.detectAndCompute(self.frame1.image)
            self.frame1.set_features(self.feats1)
            matches = self.matcher.match(self.feats0, self.feats1)
            kpts0 = self.feats0['keypoints'][matches['matches'][:,0]] 
            kpts1 = self.feats1['keypoints'][matches['matches'][:,1]]
            
            map_pts = self.frame0.map_points
            #const vel assumption
            self.frame1.set_pose(self.frame0.pose)

           


            pts_3d_ids, feat_ids = match_3D_to_2D(self.frame1, map_pts, 
                                                self.max_reproj_distance, 
                                                self.desc_norm_type, 
                                                self.max_descriptor_distance)
            map_pts = [map_pts[i] for i in pts_3d_ids]
            
            mask1, pts2d_1 = self.frame1.project_points_to_frame(np.array([p.coordinates for p in map_pts]))

            viz_img = draw_keypoints(self.frame1.image.copy(), pts2d_1, color=(0,0,255))
            viz_img = draw_keypoints(viz_img, kpts1[feat_ids], color=(0,255,0))
            feats = kpts1[feat_ids]
            #draw lines between reprojected and keypoints
            for (i,j) in zip (pts2d_1, feats):
                cv2.line(viz_img, (int(i[0]), int(i[1])), 
                                (int(j[0]), int(j[1])), color=(0,0,0), thickness=1, lineType=cv2.LINE_AA)
            
                           
            
            cv2.imshow('viz', viz_img)
            cv2.waitKey(0)

            #map_pts_world = np.array([p.coordinates for p in map_pts])
            # print('map pts world ', map_pts_world[:5])
            # map_pts_cam = self.frame1.pose[:3,:].dot(np.vstack((map_pts_world.T, np.ones((1, map_pts_world.shape[0]))))).T
            # print('map pts cam ', map_pts_cam[:5])
            
            # #if(self.camera.D.sum() == 0):
            # _, rvec, tvec = cv2.solvePnP(map_pts_cam, kpts1[feat_ids], self.camera.K, None)
            # print('rvec ', rvec)
            # print('tvec ', tvec)
            # #else:
            # #    _, rvec, tvec, _ = cv2.solvePnPRansac(map_pts_cam, kpts1[feat_ids], self.camera.K, self.camera.D)
            
            # R = cv2.Rodrigues(rvec)[0]
            # t = tvec
            # self.R = R.dot(self.R)
            # self.t = self.t + self.R.dot(t)
            # self.frame1.set_pose_from_Rt(R.T, np.matmul(-R.T, t))
            # print('pose frame1 after ', self.frame1.pose)
            #TODO: Triangulate all new feature matches between frame0 and frame1

        self.frame0 = self.frame1 
        self.index += 1
        return self.R, self.t
            
    
    
if __name__ == '__main__':
    from modvo.detectors import orb, superpoint
    from modvo.matchers.bf import BFMatcher
    from modvo.dataloaders.kitti import KITTILoader
    from modvo.utils.viz import *
    from modvo.gui.viewer import GUIDrawer
    from modvo.maps.kf_based import Frame

    dlparams = {'root_path': '/media/hudson/9708e369-632b-44b6-8c81-cc636dfdf2f36/home/hudson/Desktop/Unicamp/Doutorado/Projeto/datasets/kitti',
                'start_frame': 0,
                'stop_frame': 800,
                'sequence_name': '03',
                'camera_id': '0',}
    dataloader = KITTILoader(**dlparams)

    det_params = {'nfeatures': 1000,
                  'scaleFactor': 1.2,
                  'nlevels': 8}
    det = orb.ORBDetector(**det_params)
    # det_params = {
    #     'descriptor_dim': 256,
    #     'nms_radius': 4,
    #     'keypoint_threshold': 0.005,
    #     'max_keypoints': -1,
    #     'remove_borders': 4,
    #     'path': '../thirdparty/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth',
    #     'device': 'cuda'
    # }
    # det = superpoint.SuperPointDetector(**det_params)

    matcher_params = {'normType':cv2.NORM_L1, 
                      'crossCheck': False}
    matcher = BFMatcher(**matcher_params)
    
    vo_params = {'camera': dataloader.camera,
                 'detector': det,
                 'matcher': matcher,
                'max_reproj_distance': 2,
                'min_matches_projection': 10,
                'desc_norm_type': cv2.NORM_HAMMING,
                'max_descriptor_distance': 10.0}
    
    vo = VOLocalOptimization(**vo_params)
    gui = GUIDrawer()
    frames = []

    for i, img  in enumerate(dataloader):
        if(i > 2):
            while True:
                a=1
        print(i,'/', len(dataloader))
        R, t = vo.track(img)
        print('R ', R)
        print('t ', t)
        f = Frame(img)
        frame_pose = np.eye(4)
        frame_pose[:3,:3] = R
        frame_pose[:3,3] = t.flatten()
        f.pose = frame_pose
        frames.append(f)
        gui.draw_trajectory(frames)
        gui.draw_map_points(vo.map.get_points())
