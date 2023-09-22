import os
import cv2
from modvo.dataloaders.dataloader import DataLoader
from modvo.cameras.pinhole import PinholeCamera

class ROSStreamLoader(DataLoader):
    def __init__(self, **params):
        self.root_path = None
        self.set_camera()

    def __next__(self):
        if(self.index > self.size):
            raise StopIteration
        else:
            file_name = os.path.join(self.root_path, 'sequences', self.sequence,
                                 self.camera_id, str(self.index).zfill(6)+'.png')
            self.index += 1
            return cv2.imread(file_name)
    

    def get_timestamp(self):
        ts_path =  os.path.join(self.root_path, 'sequences', self.sequence, 'times.txt')
        lines = open(ts_path).read().split()
        return lines[self.index - 1]


    def set_camera(self):
        calib_path =  os.path.join(self.root_path, 'sequences', self.sequence, 'calib.txt')
        lines = open(calib_path).read().split()
        cam_params = {'width': 1242.0,
                      'height': 375.0,
                      'fx': float(lines[1]), 
                      'fy': float(lines[6]), 
                      'cx': float(lines[3]),
                      'cy': float(lines[7]), 
                      'k1': 0,
                      'k2': 0,
                      'p1': 0,
                      'p2': 0,
                      'k3': 0}
        self.camera = PinholeCamera(**cam_params)
      
    def get_camera(self):
        return super().get_camera()

if __name__ == '__main__':
    dlparams = {'root_path': '/root/datasets/kitti',
                'start_frame': 0,
                'stop_frame': 800,
                'sequence_name': '03',
                'camera_id': '0',}
    dataloader = KITTILoader(**dlparams)

    for i, img  in enumerate(dataloader):
        print(i,'/', len(dataloader))
