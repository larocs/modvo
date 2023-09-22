import os
import importlib
import argparse
import yaml


def main(args):
    with open(args.pipeline_config, 'r') as f:
        config = yaml.safe_load(f)

    #loading classes
    dloader_class = config['dataloader']['class']
    module = importlib.import_module('modvo.dataloaders.'+dloader_class.rsplit('.',1)[0])
    attr = getattr(module, dloader_class.rsplit('.', 1)[-1])
    dataloader = attr(**config['dataloader'])
    
    det_class = config['detector']['class']
    module = importlib.import_module('modvo.detectors.'+det_class.rsplit('.', 1)[0])
    attr = getattr(module, det_class.rsplit('.', 1)[-1])
    detector = attr(**config['detector'])

    mat_params = {'camera': dataloader.get_camera()}
    config['matcher'].update(mat_params) 
    mat_class = config['matcher']['class']
    module = importlib.import_module('modvo.matchers.'+mat_class.rsplit('.', 1)[0])
    attr = getattr(module, mat_class.rsplit('.', 1)[-1])
    matcher = attr(**config['matcher'])

    voparams = {'camera': dataloader.get_camera(),
                'detector': detector,
                'matcher': matcher}
    config['vo'].update(voparams)  
    vo_class = config['vo']['class']
    module = importlib.import_module('modvo.vo.'+vo_class.rsplit('.', 1)[0])
    attr = getattr(module, vo_class.rsplit('.', 1)[-1])
    vo = attr(**config['vo'])

    os.makedirs(args.output_path, exist_ok=True)
    log_fopen = open(os.path.join(args.output_path, args.trajectory_file), mode='a')

    for i, img  in enumerate(dataloader):
        print(i,'/', len(dataloader))
        R, t = vo.track(img)
        
        print(R[0, 0], R[0, 1], R[0, 2], t[0, 0],
              R[1, 0], R[1, 1], R[1, 2], t[1, 0],
              R[2, 0], R[2, 1], R[2, 2], t[2, 0],
              file=log_fopen)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_config', type=str, help='Path to the pipeline configuration file')
    parser.add_argument('--output_path', type=str, default = '/root/modvo/results/', help='path to save all outputs')
    parser.add_argument('--trajectory_file', type=str, default = 'trajectory.txt', help='name of the trajectory file')
    #parser.add_argument('--save_kpts_viz', action='store_true', help='save keypoints visualization')
    #parser.add_argument('--save_match_viz', action='store_true', help='save matching visualization')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)




    