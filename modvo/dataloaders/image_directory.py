import os
import yaml
import cv2
from modvo.dataloaders.dataloader import DataLoader
from modvo.cameras.pinhole import PinholeCamera

class ImageDirectoryLoader(DataLoader):
    '''
        Loader for images stored in a directory considering that the images names are their timestamps
    '''
    def __init__(self, **params):
        self.root_path = params['root_path']
        self.start = params['start_frame']
        self.stop = params['stop_frame']
        self.size = params['stop_frame']-params['start_frame']
        self.folder = params['folder_name']
        self.type = 'dataset'
        self.index = 0
        self.set_camera()
        self.is_running = True
        self.images_names = sorted(os.listdir(os.path.join(self.root_path, self.folder)))
        print(self.images_names)
    def __next__(self):
        if(self.index > self.size):
            self.is_running = False
            raise StopIteration
        else:
            file_name = os.path.join(self.root_path, self.folder, self.images_names[self.index])
            self.index += 1
            return cv2.imread(file_name)
    
    def get_timestamp(self):
        return self.images_names[self.index].split('.')[0]

    def set_camera(self):
        calib_path =  os.path.join(self.root_path, 'calib.yaml')
        #read yaml file
        with open(calib_path, 'r') as stream:
            try:
                calib = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        cam_params = {'width': calib['image_width'],
                      'height': calib['image_height'],
                      'fx': calib['camera_matrix']['fx'],
                      'fy': calib['camera_matrix']['fy'],
                      'cx': calib['camera_matrix']['cx'],
                      'cy': calib['camera_matrix']['cy'],
                      'k1': calib['distortion_coefficients']['k1'],
                        'k2': calib['distortion_coefficients']['k2'],
                        'p1': calib['distortion_coefficients']['p1'],
                        'p2': calib['distortion_coefficients']['p2'],
                        'k3': calib['distortion_coefficients']['k3']}

        self.camera = PinholeCamera(**cam_params)
      
    def get_camera(self):
        return super().get_camera()

if __name__ == '__main__':
    dlparams = {'root_path': '/root/dummy_dataset',
                'folder_name': 'images',
                'start_frame': 0,
                'stop_frame': 800,
                }
    dataloader = ImageDirectoryLoader(**dlparams)
    for i, img  in enumerate(dataloader):
        print(i,'/', len(dataloader))