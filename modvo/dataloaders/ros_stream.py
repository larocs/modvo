#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from threading import Thread

from multiprocessing import Process, Queue
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
        self.buffer = Queue(self.buffer_size)
        self.camera = None
        self.rate = rospy.Rate(self.frame_rate)
        self.is_running = False
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
        image = self.bridge.compressed_imgmsg_to_cv2(image)
        self.buffer.put(image)
        if self.buffer.full():
            self.buffer.get()

    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.buffer.empty():
            return self.buffer.get()
        else:
            return None
    
    def finish(self):
        self.is_running = False

    def run(self):
        print('Running ROS Stream Dataloader')
        self.is_running = True
        while not rospy.is_shutdown():
            self.rate.sleep()
        self.finish()
        print('ROS Stream Dataloader finished')
        
    def get_timestamp(self):
        return rospy.Time.now().to_sec()