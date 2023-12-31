#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from threading import Thread
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
        self.camera_info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber(rgb_topic, CompressedImage, self.image_callback)
        
        self.bridge = CvBridge()
        self.buffer = []
        self.camera = None
        self.rate = rospy.Rate(self.frame_rate)
        self.is_running = False
        self.index = 0
        print('Waiting for Camera Info Topic...')
        rospy.wait_for_message(cam_info_topic, CameraInfo, timeout=10)

        self.loader_thread = Thread(target = self.run, daemon=True)
        self.loader_thread.start()
  
    def camera_info_callback(self, cam_info):
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

    def image_callback(self, image):
        self.index += 1
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

    def run(self):
        print('Running ROS Stream Dataloader')
        self.is_running = True
        while not rospy.is_shutdown():
            self.rate.sleep()
        self.is_running = False
        print('ROS Stream Dataloader finished')
        
    def get_timestamp(self):
        return rospy.Time.now().to_sec()