import rospy
import roslib
import sys
import cv2
import time
import numpy as np
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import threading

# ros communications
rospy.init_node('usb_cam_node') # , anonymous=True
image_pub_L = rospy.Publisher("/camera1/usb_cam1/image_raw", Image, queue_size=10)

frequency = 50 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)

bridge = CvBridge()

# set video id
cap_L = cv2.VideoCapture(0)
cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920
height = 1080
cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
t2 = time.time()
# image_L = np.array((height, width,  3))

def run():
    while not rospy.is_shutdown():
        print('opencv version: ', cv2.__version__)
        # get a image
        t1 = time.time()
        ret_L, image_L = cap_L.read()
        # if image_L is not None:
        #     t3 = time.time()
        #     # print('None->img time: ', t3-t2)
        #     Limg_ROI = image_L[3:1080, 258:1661]
        #
        #     rz_L_Img_1080p = cv2.resize(Limg_ROI, (1920, 1080))
        #     # cv2.imshow('Img_Right', image_R)
        #     # cv2.imshow('Img_Left', image_L)
        #     # cv2.imshow('Img_LR', rz_LR_Img)
        #     try:
        #         t4 = time.time()
        #         ros_LImg = bridge.cv2_to_imgmsg(image_L, "bgr8")
        #         ros_LImg.header.stamp = rospy.Time.now()
        #         # ros_RImg = bridge.cv2_to_imgmsg(image_R, "bgr8")
        #         # ros_RImg.header.stamp = rospy.Time.now()
        #         # ros_LRImg = bridge.cv2_to_imgmsg(rz_LR_Img, "bgr8")
        #         # ros_LRImg.header.stamp = rospy.Time.now()
        #         # ros_LRImg_1080p = bridge.cv2_to_imgmsg(rz_LR_Img_1080p, "bgr8")
        #         # ros_LRImg_1080p.header.stamp = rospy.Time.now()
        #         image_pub_L.publish(ros_LImg)
        #         # image_pub_R.publish(ros_RImg)
        #         # image_pub_LR.publish(ros_LRImg)
        #         # image_pub_LR_1080p.publish(ros_LRImg_1080p)
        #         # loop_rate.sleep()
        #         t5 = time.time()
        #         # print('publish time: ', t5-t4)
        #     except CvBridgeError as e:
        #         print(e)
        #     t2 = time.time()
        #
        # # if cv2.waitKey(1) & 0xFF == ord('q'):
        # #     break
        # # if cv2.waitKey(1) & 0xFF == ord('s'):
        # #     cv2.imwrite('L-test.png', image_L)
        # t2 = time.time()
        print('Done.', (t2 - t1), 'sec')

fixed_size = 10 * 1024 * 1024
threading.stack_size(fixed_size)
thread = threading.Thread(target=run) #, args=(cap_L, )
thread.start()
cap_L.release()
cv2.destroyAllWindows()
