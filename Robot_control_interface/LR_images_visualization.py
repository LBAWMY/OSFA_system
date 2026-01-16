import rospy
import roslib
import sys
import cv2
import time
import numpy as np
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import threading

GUIimg_LR_src = None
bridge = CvBridge()

def GUIimage_LRCallback(img):
    global GUIimg_LR_src
    GUIimg_LR_src = bridge.imgmsg_to_cv2(img, "bgr8")

def thread_job(self):
    rospy.spin()

rospy.init_node('vis_node', anonymous=True)
# capture image
rospy.Subscriber("/camera1_2/gui_cam1_2/image_1080p", Image, GUIimage_LRCallback) #, queue_size=1, buff_size=2**25

frequency = 150 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)

add_thread = threading.Thread(target=thread_job)
add_thread.start()

while not rospy.is_shutdown():
    try:
        # print(type(GUIimg_LR_src))
        W, H = GUIimg_LR_src.shape[:2]
        L_Img = GUIimg_LR_src[:, :int(H/2), :]
        R_Img = GUIimg_LR_src[:, int(H/2):, :]

        cv2.imshow('Img_Right', R_Img[:,:,::-1])
        cv2.imshow('Img_Left', L_Img[:,:,::-1])

        loop_rate.sleep()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except TypeError:
        print('no data from camera')
        continue
    except AttributeError:
        print('no data from camera')
        continue
    except cv2.error:
        print('cv2 error')
        continue
    # except IndexError:
    #     print('Index error: no matching!')
    #     continue
    else:
        continue

cv2.destroyAllWindows()

