import os
import sys
import importlib
import argparse
import yaml

def main(args):
    with open(args.pipeline_config, 'r') as f:
        config = yaml.safe_load(f)

    #loading classes
    dloader_class = config['dataloader']['class']
    print('Dataloader %s' % dloader_class)
    module = importlib.import_module('modvo.dataloaders.'+dloader_class.rsplit('.',1)[0])
    attr = getattr(module, dloader_class.rsplit('.', 1)[-1])
    #get params without class name
    params = {k: v for k, v in config['dataloader'].items() if k != 'class'}
    dataloader = attr(**params)
    
    det_class = config['detector']['class']
    print('Detector %s' % det_class)
    module = importlib.import_module('modvo.detectors.'+det_class.rsplit('.', 1)[0])
    attr = getattr(module, det_class.rsplit('.', 1)[-1])
    params = {k: v for k, v in config['detector'].items() if k != 'class'}
    detector = attr(**params)

    mat_class = config['matcher']['class']
    print('Matcher %s' % mat_class)
    module = importlib.import_module('modvo.matchers.'+mat_class.rsplit('.', 1)[0])
    attr = getattr(module, mat_class.rsplit('.', 1)[-1])
    params = {k: v for k, v in config['matcher'].items() if k != 'class'}
    matcher = attr(**params)
    if(dataloader.get_camera() is None):
        print('Dataloader camera not found')
        sys.exit(0)
    voparams = {'camera': dataloader.get_camera(),
                'detector': detector,
                'matcher': matcher}
    config['vo'].update(voparams)  
    vo_class = config['vo']['class']
    print('VO %s' % vo_class)
    module = importlib.import_module('modvo.vo.'+vo_class.rsplit('.', 1)[0])
    attr = getattr(module, vo_class.rsplit('.', 1)[-1])
    params = {k: v for k, v in config['vo'].items() if k != 'class'}
    vo = attr(**params)

    os.makedirs(args.output_path, exist_ok=True)
    log_fopen = open(os.path.join(args.output_path, args.trajectory_file), mode='a')
    print('Enable GUI: ', args.enable_gui)
    print('Save keypoints: ', args.save_keypoints)

    if args.enable_gui:
        import numpy as np
        from modvo.maps.kf_based import Frame
        from modvo.gui.viewer import GUIDrawer
        drawer = GUIDrawer()
        frames = []
    
    if args.save_keypoints:
        import utils.viz as viz
        os.makedirs(os.path.join(args.output_path, 'keypoints'), exist_ok=True)

    while dataloader.is_running:
        image = next(dataloader)
        if(image is None):
            continue
        print('img shape ', image.shape)
        R, t = vo.track(image)
        if args.enable_gui:
            f = Frame(image)
            frame_pose = np.eye(4)
            frame_pose[:3,:3] = R
            frame_pose[:3,3] = t.flatten()
            f.pose = frame_pose
            frames.append(f)
            drawer.draw_trajectory(frames)
        if(dataloader.type == 'dataset'):
            i = dataloader.index
            print(i,'/', len(dataloader))
        else:
            print('frame ', dataloader.index)
        
        print(R[0, 0], R[0, 1], R[0, 2], t[0, 0],
              R[1, 0], R[1, 1], R[1, 2], t[1, 0],
              R[2, 0], R[2, 1], R[2, 2], t[2, 0],
              file=log_fopen)

        if args.save_keypoints:
            feats = detector.detectAndCompute(image)
            image_kpts = viz.draw_keypoints(image, feats['keypoints'])
            viz.save_image(image_kpts, os.path.join(args.output_path, 'keypoints', str(dataloader.index)+'.png'))
          
    print("Exiting...")
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_config', type=str, help='Path to the pipeline configuration file')
    parser.add_argument('--output_path', type=str, default = '/root/modvo/results/', help='path to save all outputs')
    parser.add_argument('--trajectory_file', type=str, default = 'trajectory.txt', help='name of the trajectory file')
    parser.add_argument('--enable_gui', action='store_true', help='use this flag to enable gui')
    parser.add_argument('--save_keypoints', action='store_true', help='use this flag to save images with keypoints')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)