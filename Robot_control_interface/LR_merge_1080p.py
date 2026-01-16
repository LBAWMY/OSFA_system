import rospy
import roslib
import sys
import cv2
import time
import numpy as np
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

img_L_src = None
img_R_src = None
bridge = CvBridge()

def image_LCallback(img):
    global img_L_src
    img_L_src = bridge.imgmsg_to_cv2(img, "bgr8")

def image_RCallback(img):
    global img_R_src
    img_R_src = bridge.imgmsg_to_cv2(img, "bgr8")

rospy.init_node('merge_image_node', anonymous=True)
frequency = 50 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)
# capture image
# rospy.Subscriber("/camera1/usb_cam1/image_raw", Image, image_LCallback, queue_size=1, buff_size=2**1000)
# rospy.Subscriber("/camera2/usb_cam2/image_raw", Image, image_RCallback, queue_size=1, buff_size=2**1000)
rospy.Subscriber("/camera1/usb_cam1/image_raw", Image, image_LCallback)
rospy.Subscriber("/camera2/usb_cam2/image_raw", Image, image_RCallback)
image_pub_LR = rospy.Publisher("/camera1_2/usb_cam1_2/image_raw", Image, queue_size=1)
image_pub_LR_1080p = rospy.Publisher("/camera1_2/usb_cam1_2/image_raw_1080p", Image, queue_size=1)

while not rospy.is_shutdown():
    if img_L_src is not None and img_R_src is not None:
        img_L_rsz = cv2.resize(img_L_src, (640, 480))
        img_R_rsz = cv2.resize(img_R_src, (640, 480))

        img_R_rsz[:,:,0] = img_R_rsz[:,:,0]/1.1
        img_R_rsz[:,:,1] = img_R_rsz[:,:,1]/1.4
        img_R_rsz[:,:,2] = img_R_rsz[:,:,2]/1.05
        img_R_rsz[:,:,0][img_R_rsz[:,:,0]>15] -= 5
        img_R_rsz[:,:,1][img_R_rsz[:,:,1]>10] -= 5
        img_R_rsz[:,:,2][img_R_rsz[:,:,2]<240] += 15
        img_R_rsz[:,:,2][img_R_rsz[:,:,2]<100] -= 15

        Img_LR_rsz = cv2.hconcat([img_L_rsz, img_R_rsz])
        cv2.imshow('Img_LR', Img_LR_rsz[:,:,:])

        Img_LR_1080p = cv2.hconcat([img_L_src, img_R_src])
        # cv2.imshow('Img_Right', Img_LR_1080p[:,:,:])

        try:
        # print(type(GUIimg_LR_src))
            ros_LRImg_1080p = bridge.cv2_to_imgmsg(Img_LR_1080p, "bgr8")
            ros_LRImg_1080p.header.stamp = rospy.Time.now()
            image_pub_LR_1080p.publish(ros_LRImg_1080p)

            ros_LRImg_rsz = bridge.cv2_to_imgmsg(Img_LR_rsz, "bgr8")
            ros_LRImg_rsz.header.stamp = rospy.Time.now()
            image_pub_LR.publish(ros_LRImg_rsz)
        except CvBridgeError as e:
            print(e)

    loop_rate.sleep()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
