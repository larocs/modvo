#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from modvo.dataloaders.dataloader import DataLoader
from modvo.cameras.pinhole import PinholeCamera

class ROSStreamLoader(DataLoader):
    def __init__(self, **params):
        rospy.init_node('ros_stream_dataloader', anonymous=True)
        self.buffer_size = params['buffer_size']
        self.frame_rate = params['frame_rate']
        rgb_topic = params['rgb_topic']
        cam_info_topic = params['cam_info_topic']
        self.type = 'stream'
        
        self.image_sub = rospy.Subscriber(rgb_topic, CompressedImage, self.callback)
        
        self.bridge = CvBridge()       
        self.buffer = []
        cam_params = {'width': 1280,
                      'height': 720,
                      'fx': 912.469360351562, 
                      'fy': 912.747497558594, 
                      'cx': 638.081298828125,
                      'cy': 912.469360351562, 
                      'k1': 0,
                      'k2': 0,
                      'p1': 0,
                      'p2': 0,
                      'k3': 0}
        self.camera = PinholeCamera(**cam_params)
        self.rate = rospy.Rate(self.frame_rate)
        self.is_running = False
        
    def callback(self, image):
        image = self.bridge.compressed_imgmsg_to_cv2(image)
        self.buffer.append(image)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.buffer) > 0:
            return self.buffer.pop(0)
        else:
            return None
    
    def finish(self):
        self.is_running = False
        self.image_sub.unregister()

    def run(self):
        self.is_running = True
        rospy.on_shutdown(self.finish)
        while not rospy.is_shutdown():
            self.rate.sleep()
        self.is_running = False
    
    def get_timestamp(self):
        return rospy.Time.now().to_sec()