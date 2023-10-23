#!/usr/bin/env python3

import rospy
import numpy as np
from pal.products.qcar import QCarRealSense
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge


class RGBStreamingNode:
    def __init__(self):
        rospy.init_node('rbg_streaming_node')

        # RGB camera parameters
        width  = 1280
        height = 720
        rate = 30

        self.rate = rospy.Rate(rate)

        # Initialize ros publishers
        self.imagePub = rospy.Publisher('/qcar/rgb/compressed', CompressedImage, queue_size=10)
        self.camInfoPub = rospy.Publisher('/qcar/rgb/camera_info', CameraInfo, queue_size=10)
        self.bridge = CvBridge()
        self.rgb = QCarRealSense(mode='RGB',
                                frameWidthRGB    = width,
                                frameHeightRGB   = height,
                                frameRateRGB	 = rate,)

        self.intrinsics = [912.469360351562, 0, 638.081298828125,
                           0, 912.747497558594, 350.80615234375,
                           0, 0, 1]
        self.distortion = [0., 0., 0., 0., 0.]

    def run(self):
        rospy.on_shutdown(self.terminate_camera)
        while not rospy.is_shutdown():
            self.rgb.read_RGB()
            self.publish_image()
            self.publish_cam_info()
            self.rate.sleep()
                      
    def publish_image(self):
        img_msg = CompressedImage()
        img_msg.data = self.bridge.cv2_to_compressed_imgmsg(self.rgb.imageBufferRGB)
        img_msg.header.stamp =  rospy.Time.now()
        img_msg.header.frame_id = 'camera_link'
        self.imagePub.publish(img_msg)


    def publish_cam_info(self):
        cam_info_msg = CameraInfo()
        cam_info_msg.header.stamp = rospy.Time.now()
        cam_info_msg.header.frame_id = 'camera_link'
        cam_info_msg.width = self.rgb.frameWidthRGB
        cam_info_msg.height = self.rgb.frameHeightRGB
        cam_info_msg.distortion_model = 'plumb_bob'
        cam_info_msg.K = self.intrinsics
        cam_info_msg.D  = self.distortion
        self.camInfoPub.publish(cam_info_msg)
    
    def terminate_camera(self):
        self.rgb.terminate()


def main():
    node = RGBStreamingNode()
    node.run()

if __name__ == '__main__':
	main()