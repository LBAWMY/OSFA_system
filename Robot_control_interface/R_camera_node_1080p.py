import rospy
import roslib
import sys
import cv2
import time
import numpy as np
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

# def publish_image(imgdata):
#     image_temp=Image()
#     header = Header(stamp=rospy.Time.now())
#     header.frame_id = 'map'
#     image_temp.encoding='rgb8'
#     image_temp.data=np.array(imgdata).tostring()
#     #print(imgdata)
#     #image_temp.is_bigendian=True
#     image_temp.header=header
#     image_temp.step=1241*3
#     image_pubulish.publish(image_temp)

# ros communications
rospy.init_node('usb_cam2_node') # , anonymous=True
image_pub_R = rospy.Publisher("/camera2/usb_cam2/image_raw", Image)

frequency = 50 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)

bridge = CvBridge()

# set video id
cap_R = cv2.VideoCapture(1)
cap_R.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_YUYV) # cv2.CAP_MODE_YUYV
# cap_R.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# width = 1920
# height = 1080
# cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
width = 720
height = 576
cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while not rospy.is_shutdown():
    print('opencv version: ', cv2.__version__)
    # get a image
    t1 = time.time()
    ret_R, image_R = cap_R.read()
    t1_2 = time.time()

    print('right img read. ', (t1_2 - t1), 'sec')
    if image_R is not None:
        t2_3 = time.time()
        Rimg_ROI = image_R[0:1061, 47:1853]

        # rz_R_Img = cv2.resize(Rimg_ROI, (640, 480))
        # rz_L_Img = cv2.resize(Limg_ROI, (640, 480))
        # rz_LR_Img = cv2.hconcat([rz_L_Img, rz_R_Img])

        rz_R_Img_1080p = cv2.resize(Rimg_ROI, (1920, 1080))

        t2 = time.time()
        print('img manipulation.', (t2 - t2_3), 'sec')
        print('right FrameRate: ', cap_R.get(cv2.CAP_PROP_FPS))
        # cv2.imshow('Img_Right', image_R)
        # cv2.imshow('Img_Left', image_L)
        cv2.imshow('Img_LR', rz_R_Img_1080p)
        try:
            # ros_LImg = bridge.cv2_to_imgmsg(image_L, "bgr8")
            # ros_LImg.header.stamp = rospy.Time.now()
            ros_RImg = bridge.cv2_to_imgmsg(rz_R_Img_1080p, "bgr8")
            ros_RImg.header.stamp = rospy.Time.now()
            # ros_LRImg = bridge.cv2_to_imgmsg(rz_LR_Img, "bgr8")
            # ros_LRImg.header.stamp = rospy.Time.now()
            # ros_LRImg_1080p = bridge.cv2_to_imgmsg(rz_LR_Img_1080p, "bgr8")
            # ros_LRImg_1080p.header.stamp = rospy.Time.now()
            # image_pub_L.publish(ros_LImg)
            image_pub_R.publish(ros_RImg)
            # image_pub_LR.publish(ros_LRImg)
            # image_pub_LR_1080p.publish(ros_LRImg_1080p)
            # rospy.sleep()
        except CvBridgeError as e:
            print(e)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('R-test.png', image_R)
    t3 = time.time()
    print('Done.', (t3 - t1), 'sec')

cap_R.release()
cv2.destroyAllWindows()
