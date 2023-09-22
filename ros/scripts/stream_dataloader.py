#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from modvo.dataloaders.dataloader import DataLoader
from modvo.cameras.pinhole import PinholeCamera

class ROSStreamLoader(DataLoader):
    def __init__(self, **params):
        self.buffer_size = params['buffer_size']
        self.type = 'stream'
        image_sub = message_filters.Subscriber('rgb/compressed', CompressedImage)
        cam_info_sub = message_filters.Subscriber('rgb/camera_info', CameraInfo)
        self.ts = message_filters.TimeSynchronizer([image_sub, cam_info_sub], 10)
        self.ts.registerCallback(self.callback)
        self.bridge = CvBridge()       
        self.buffer = []

    def callback(self, image, cam_info):
        image = self.bridge.compressed_imgmsg_to_cv2(image)
        cam_params = {'width': cam_info.width,
                      'height': cam_info.height,
                      'fx': cam_info.K[0], 
                      'fy': cam_info.K[4], 
                      'cx': cam_info.K[2],
                      'cy': cam_info.K[5], 
                      'k1': 0,
                      'k2': 0,
                      'p1': 0,
                      'p2': 0,
                      'k3': 0}
        self.camera = PinholeCamera(**cam_params)
        self.buffer.append(image)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.buffer.pop(0)
    
    def run(self):
        while not rospy.is_shutdown():
                rospy.sleep(0.01)
    

def main():
    rospy.init_node('ros_stream_dataloader', anonymous=True)
    loader = ROSStreamLoader(buffer_size=10)
    loader.run()


if __name__ == '__main__':
    main()