import cv2
import numpy as np
from modvo.vo.tracker import Tracker
from modvo.utils.geometry import pose_from_kpts, triangulate_points, match_3D_to_2D
from modvo.maps.kf_based import Frame, KFBasedMap
from modvo.optimizers.g2o import G2OOptimizer


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
        self.vel = np.array([0, 0, 1.0])      

    def track(self, image):
        frame = Frame(image)
        frame.set_camera(self.camera)
        frame.detector = self.detector
        feats = self.detector.detectAndCompute(image)
        frame.set_features(feats)
        self.map.add_frame(frame)
        frame.is_keyframe = True #all frames are keyframes for now
        self.frame1 = frame

        if(self.index == 0):
            #First frame
            self.R = np.eye(3)
            self.t = np.zeros((3,1))


        elif(self.index == 1):
            self.feats0 = self.frame0.features
            self.feats1 = self.frame1.features
            matches = self.matcher.match(self.feats0, self.feats1)
            
            #get matched kpts
            kpts0 = self.frame0.keypoints[matches['matches'][:,0]]
            kpts1 = self.frame1.keypoints[matches['matches'][:,1]]
            
            mask, R, t, self.n_inliers = pose_from_kpts(kpts0, kpts1, self.camera)
            inliers_mask = (mask.ravel() == 1)
            self.t = self.t + self.R.dot(t)
            self.R = R.dot(self.R)
            #set frame pose
            self.frame1.set_pose_from_Rt(R.T, np.matmul(-R.T, t))

            #triangulate points
            in_kpts0un = self.frame0.keypoints_un[matches['matches'][:,0]][inliers_mask]
            in_kpts1un = self.frame1.keypoints_un[matches['matches'][:,1]][inliers_mask]
            points3D = triangulate_points(in_kpts0un.T, in_kpts1un.T, self.frame0.pose[:3,:], self.frame1.pose[:3,:])

            #filter points based on depth consistency
            mask0, _ = self.frame0.project_points_to_frame(points3D)
            mask1, _ = self.frame1.project_points_to_frame(points3D)
            valid_ids = (mask0 & mask1)
            points3D = points3D[valid_ids]
            #add them to the map
            self.map.add_points_from_coordinates(points3D)
            descs1_valid = self.feats1['descriptors'][matches['matches'][inliers_mask,1]][valid_ids]
            self.frame1.set_map_points_and_descs(self.map.get_points(), descs1_valid)
        else:
            map_pts = self.map.points
            new_R = self.frame0.pose[:3,:3]
            new_t = self.frame0.pose[:3,3]
            new_t[2]+=0.9996969
            new_pose = np.eye(4)
            new_pose[:3,:3] = new_R
            new_pose[:3,3] = new_t
            self.frame1.set_pose(new_pose)

            pts_3d_ids, _ = match_3D_to_2D(self.frame1, map_pts, 
                                            self.max_reproj_distance, 
                                            self.desc_norm_type, 
                                            self.max_descriptor_distance)
            
            map_pts = [map_pts[i] for i in pts_3d_ids]

            self.frame1.set_map_points_and_descs(map_pts, [pt.get_descriptor() for pt in map_pts])
            
            #Triangulate all new feature matches between frame0 and frame1
            matches = self.matcher.match(self.frame0.features, self.frame1.features)
            #get matched kpts
            in_kpts0un = self.frame0.keypoints_un[matches['matches'][:,0]]
            in_kpts1un = self.frame1.keypoints_un[matches['matches'][:,1]]
            points3D = triangulate_points(in_kpts0un.T, in_kpts1un.T, self.frame0.pose[:3,:], self.frame1.pose[:3,:])
            #filter points based on depth consistency
            mask0, _ = self.frame0.project_points_to_frame(points3D)
            mask1, _ = self.frame1.project_points_to_frame(points3D)
            valid_ids = (mask0 & mask1)
            points3D = points3D[valid_ids]

            #add them to the map if they are not already there
            new_pts_mask = self.map.check_new_points(points3D)
            points3D = points3D[new_pts_mask]
            idxs = self.map.add_points_from_coordinates(points3D)
            
            #add new points to frame1
            new_pts = [self.map.get_point(i) for i in idxs]
            new_descs = self.feats1['descriptors'][matches['matches'][:,1]][valid_ids][new_pts_mask]
            self.frame1.set_map_points_and_descs(new_pts, new_descs)

            n_inliers = self.optimizer.optimize(self.frame1)
            print(f'Optimized pose with {n_inliers} inliers')
            

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
                'max_descriptor_distance': 100.0}
    
    vo = VOLocalOptimization(**vo_params)
    gui = GUIDrawer()
    frames = []

    for i, img  in enumerate(dataloader):
        if(i > 10):
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
        gui.draw_map(frames, vo.map.get_points())
        print('number of points ', len(vo.map.get_points()))