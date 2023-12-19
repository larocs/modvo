import g2o
import numpy as np
from modvo.optimizers.optimizer import Optimizer

class G2OOptimizer(Optimizer):
    def __init__(self):
        self.pose_opt_min_correspondences = 3
        self.pose_opt_chi2 = [9.210,7.378,5.991,5.991]
        self.pose_opt_iterations = [10, 10, 7, 5]
    
    def optimize(self, frame):
        optimizer = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())       
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        optimizer.set_algorithm(solver)
        
        #set frame vertex
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(g2o.SE3Quat(frame.pose[:3,:3], frame.pose[:3,3]))
        v_se3.set_id(0)
        v_se3.set_fixed(False)
        optimizer.add_vertex(v_se3)

        #keypoints 
        kptsu = frame.camera.undistort_keypoints(frame.features['keypoints'].T).T
        scaleLevels = frame.detector.getScaleLevels()
        levelSigma2 = np.square(scaleLevels)
        invLevelSigma2 = 1.0/levelSigma2
        outliers = np.zeros(len(kptsu), dtype=bool)

        #set map point vertices
        delta = np.sqrt(5.991)
        edges, idxs = [], []
        for i, p in enumerate(frame.map_points):
            edge = g2o.EdgeSE3ProjectXYZOnlyPose()
            edge.set_vertex(0, optimizer.vertex(0))
            edge.set_measurement(kptsu[i])
            
            edge.set_information(np.eye(2)*invLevelSigma2[frame.features['octaves'][i]])
            edge.set_robust_kernel(g2o.RobustKernelHuber(delta))

            edge.fx = frame.camera.fx 
            edge.fy = frame.camera.fy
            edge.cx = frame.camera.cx
            edge.cy = frame.camera.cy
            edge.Xw = p.coordinates
            optimizer.add_edge(edge)

            edges.append(edge)
            idxs.append(i)

        if(len(edges) < self.pose_opt_min_correspondences):
            print(f'Pose optimization error: not enough correspondences - {len(edges)}')
            return 0
        
        #Based on ORBSLAM: https://github.com/raulmur/ORB_SLAM/blob/master/src/Optimizer.cc#L239
        #Perform 4 optimizations, decreasing the inlier region
        #From second to final optimization we include only inliers in the optimization
        #At the end of each optimization we check which points are inliers

        for i in range(len(self.pose_opt_iterations)):
            optimizer.initialize_optimization()        
            optimizer.optimize(self.pose_opt_iterations[i])
            nBad = 0
            for edge, idx in zip(edges, idxs):
                if(outliers[idx]):
                    edge.compute_error()

                if(edge.chi2() > self.pose_opt_chi2[i]):
                    outliers[idx] = True
                    edge.set_level(1)
                    nBad += 1
                else:
                    outliers[idx] = False
                    edge.set_level(0)

            if(len(optimizer.edges()) < 10):
                print(f'Pose optimization error: not enough edges - {len(optimizer.edges())}')
                break

        nInliers = len(edges)-nBad
        if(nInliers == 0):
            print(f'Pose optimization: no inliers found!')
            return 0
        
        est = v_se3.estimate()
        pose = g2o.Isometry3d(est.orientation(), est.position()).matrix()
        frame.set_pose(pose)

        return nInliers