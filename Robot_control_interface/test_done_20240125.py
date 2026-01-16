# -*- coding: utf-8 -*
import rospy
import sys
import roslib
# roslib.load_manifest('ur5_endoscope_arm')
import cv2
import numpy as np
import copy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, Joy, JointState
# from ur5_endoscope_arm.msg import *
from robots.ur5_vision_msg import ur5_vision_msg

from PyQt5.QtCore import pyqtSlot, QThread
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QFont

from functools import wraps
from qt_gui.GUI_interface import Ui_MainWindow
from qt_gui.Left_1080p import Ui_Left1080p
from qt_gui.Right_1080p import Ui_Right1080p

from robots.ur5 import UR5
from backend.utils_robotics import RotZ
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64, Int32, Bool

from yolo_bbox_msg.msg import Boundingbox
from blackbox_msg.msg import Blackboxinfo

import socket
import time
import threading
import keyboard

def debug_class(calss_name='MainWindow'):
    # print('in debug_class')
    def debug(f):
        # print('in debug')
        @wraps(f)
        def print_debug(*args, **kwargs):
            # print('in print_debug')
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print(calss_name + '.' + f.__name__ + '() Error！')
                print('Error：', e)

        return print_debug

    return debug


class MainWindow_ROS(QMainWindow, Ui_MainWindow):
    @debug_class('MainWindow')
    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow_ROS, self).__init__(parent)
        self.setupUi(self)
        self.img_left_src_RGB_1080p = None
        self.show_L_Img_1080p_flag = False
        self.show_R_Img_1080p_flag = False
        # Initialization: UI
        # self.label_panel_img_show.setScaledContents(True)  # Show image with adaptive scale
        # self.img_none = np.ones((480, 640, 3), dtype=np.uint8)*255
        self.main_panel_img_H = 768
        self.main_panel_img_W = 1024
        self.main_3d_img_H = 1080 #720
        self.main_3d_img_W = 1920 #1280
        self.shift_display = 0
        self.shift_scale = 0.85
        self.shift_scale_appear = 1-(1-self.shift_scale)/2
        # self.shift_scale_appear = 0.95
        self.shift_display_x = round((1-self.shift_scale)*self.main_3d_img_W/2)
        self.shift_display_y = round((1-self.shift_scale)*self.main_3d_img_H/2)
        self.img_none = np.ones((self.main_panel_img_H, self.main_panel_img_W, 3), dtype=np.uint8) * 255

        self.img_right_src = None
        self.img_left_src_1080p = None
        self.img_right_src_1080p = None
        # --------- add channel -------
        self.img_none = add_alpha_channel(self.img_none)
        self.img_right_src_RGB = self.img_none
        # self.show_img()
        # Initialization: Network
        # ToDO
        # Other settings
        self.camera_left_index = 1
        self.video_flag = False
        # self.vision_params = ur5_vision_msg()
        self.vision_params = {}
        # self.Kc = np.array([[677.0186, 0.0, 334.5997],
        #                     [0.0, 664.2157, 231.5869],
        #                     [0.0, 0.0, 1.0]])
        # self.Kc = np.array([[614.3098, 0.0, 313.67928],
        #                     [0.0, 738.4367, 236.43232],
        #                     [0.0, 0.0, 1.0]])
        # self.Kc = np.array([[2024.8, 0.0, 1002.4],
        #                     [0, 1493.5, 517.6556],
        #                     [0.0, 0.0, 1.0]]) # curi - 1080p
        self.Kc = np.array([[1996.46, 0.0, 1037.48],
                            [0, 1508.56, 555.65],
                            [0.0, 0.0, 1.0]]) # curi Left - 1080p
        self.robot_cVc_vis = np.array([0.0, 0.0, 0.0])

        # cadaver2
        # self.Kc = np.array([[613.7328, 0.0, 319.3413],
        #                     [0.0, 736.9071, 234.2559],
        #                     [0.0, 0.0, 1.0]])

        self.NUM_TOOLS_TO_TRACK = 5

        # Initialization: ROS
        rospy.init_node("Mainwindow_node", anonymous=True)
        # capture image
        self.bridge = CvBridge()

        self.cap_L = cv2.VideoCapture(0) # 1
        self.cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        width = 1920
        height = 1080
        self.cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap_L.set(cv2.CAP_PROP_FPS, 50)
        self.img_right_src_1080p = None
        cameraL_thread = threading.Thread(target=self.cameraL_job)
        cameraL_thread.daemon = True
        cameraL_thread.start()

        keyboard_thread = threading.Thread(target=self.keyboard_job)
        keyboard_thread.daemon = True
        keyboard_thread.start()

        # capture joystick
        self.joy_src = Joy()
        self.joy_src.axes = np.zeros((3,), dtype=np.float32)
        self.joy_src.buttons = np.zeros((6,), dtype=np.bool)
        rospy.Subscriber("joy", Joy, self.joyCallback)
        # capture the ur5 joint state
        self.joint_state = None
        rospy.Subscriber("joint_states", JointState, self.jointStateCallback)
        # publish the vision params
        # self.vision_params_pub = rospy.Publisher("vision_params", ur5_vision_msg, queue_size=10)
        # publish command to control ur5
        self.ur5_crtl_script_pub = rospy.Publisher("ur_driver/URScript", String, queue_size=1)
        # publish the main tool index to the tracker: update the main tool index
        self.main_tool_id_pub = rospy.Publisher("/multitools/main_tool_id", Int32, queue_size=1)
        self.system_version_pub = rospy.Publisher("/system_info/version", Int32, queue_size=1)
        # publish the scale factor
        self.scale_bbox_pub = rospy.Publisher("/customize_settings/scale_bbox", Float64, queue_size=1)
        # publish the middle depth setting
        self.mid_depth_pub = rospy.Publisher("/customize_settings/mid_depth", Float64, queue_size=1)
        self.mid_range_pub = rospy.Publisher("/customize_settings/mid_range", Float64, queue_size=1)
        # publish the middle depth setting  todo
        self.rotation_pub = rospy.Publisher("/customize_settings/rotation", Float64, queue_size=1)
        # publish the initial laparoscope direction
        self.base_framex_pub = rospy.Publisher("/init_settings/baseframex", Vector3, queue_size=1)
        # publish the 1080p left-right images
        # self.img_1080p_pub = rospy.Publisher("/camera1_2/gui_cam1_2/image_1080p", Image, queue_size=1)
        # publish the blackbox info
        self.blackbox_pub = rospy.Publisher("/blackbox_info", Blackboxinfo, queue_size=1)
        self.blackbox_info_msg = Blackboxinfo()

        self.loop_rate = rospy.Rate(150)  # 50hz
        # subscribe the predicted action
        rospy.Subscriber("/track/pred_action", Vector3, self.predActionCallback)
        # subscribe the estimated depth
        rospy.Subscriber("/track/main_tool_depth", Float64, self.toolDepthCallback)

        self.pred_position_tools = [None] * self.NUM_TOOLS_TO_TRACK
        self.main_tool_id = 1  # by default: 0: tool1, 1:tool2, 2:tool3, 3:tool4, 4:tool5
        self.bbox_scale = 1.0  # by default: the scale is the 1.0, need to in range(0.0, 2.0)
        # self.mid_depth_init = 75  # by default: the middle depth is 75: [65, 85]
        # self.mid_range_init = 20
        # self.mid_depth = self.mid_depth_init  # by default: the middle depth is 75: [65, 85]
        # self.mid_range = self.mid_range_init
        self.mid_idx = 1
        self.mid_depth_fix = [40, 60, 80, 110]
        self.mid_range_fix = [10, 15, 25, 40]
        self.mid_depth = self.mid_depth_fix[self.mid_idx]  # by default: the middle depth is 75: [65, 85]
        self.mid_range = self.mid_range_fix[self.mid_idx]

        self.rotation = 0.0
        # button value: select the button in setting manual
        self.setting_button_index = 0  # by default: range[0, 5]
        self.SettingBoxValue = 1.0
        self.cur_time = time.time()
        self.cur_time_button = time.time()
        rospy.Subscriber("/track/pred_position_tool0", Vector3, self.tool0_predPositionCallback)
        rospy.Subscriber("/track/pred_position_tool1", Vector3, self.tool1_predPositionCallback)
        rospy.Subscriber("/track/pred_position_tool2", Vector3, self.tool2_predPositionCallback)
        rospy.Subscriber("/track/pred_position_tool3", Vector3, self.tool3_predPositionCallback)
        rospy.Subscriber("/track/pred_position_tool4", Vector3, self.tool4_predPositionCallback)
        # subsribe the refer bbox information for visulization
        self.white_bbox_refer_src = None
        rospy.Subscriber("/visualize_bbox/white_bbox", Boundingbox, self.white_referBBoxCallback)
        self.yellow_bbox_refer_src = None
        rospy.Subscriber("/visualize_bbox/yellow_bbox", Boundingbox, self.yellow_referBBoxCallback)
        self.whether_visualize = False
        self.whether_visualize_in_setting = False
        rospy.Subscriber("/visualize_bbox/whether_visualize", Bool, self.visualize_whetherBBoxCallback)

        # robot
        self.robot = None
        self.dist_r2c = -1
        self.dist_r2c_refer = -1
        self.Workspace_Inter = -1
        self.closest_dist = 40 # mm

        # gui count
        self.delay_main_panel_appear = 0.8
        self.main_panel_appear_flag = True
        self.main_panel_appear_flag_left = True
        self.main_panel_appear_flag_right = True
        self.main_panel_pure_appear_flag = True
        self.main_panel_appear_time_left = time.time()
        self.main_panel_appear_time_right = time.time()
        self.main_panel_appear_time = time.time()
        self.main_panel_pure_appear_time = time.time()
        self.uterus_panel_area_exit = True

        self.aux_control_flag = False
        self.main_control_flag = False
        self.aux_control_idx = -1

        self.delay_main_keep_small = 3.0
        # self.count_main_large = 1.0
        self.delay_main_large_to_panel = 1.5
        self.delay_panel_each = 1.0
        self.delay_panel_dorminant = 3.0
        self.delay_panel_uterus = 1.0

        self.gui_display_time = time.time()
        self.gui_display_time_delay = 0
        self.gui_main_panel_time = np.ones((1,4)) * time.time()
        self.gui_main_panel_time_delay = np.zeros((1, 4))
        self.uterus_panel_time = np.ones((1,7)) * time.time()
        self.uterus_panel_time_delay = np.zeros((1, 7))
        self.uterus_panel_time_delay_old = np.zeros((1, 7))
        self.camera_panel_time = np.ones((1,6)) * time.time()
        self.camera_panel_time_delay = np.zeros((1,6))
        self.dominant_tool_time = time.time()
        self.dominant_tool_time_delay = 0
        self.dominant_time = time.time()
        self.dominant_time_delay = 0

        # display the depth range time for lasting 1 second
        # self.vis_setting_depth_time = time.time()
        # self.vis_setting_depth_time_delay = 2

        # load gui sources
        self.alpha_main_small_bg = 0.6
        self.alpha_main_small_gui = 0.75
        self.alpha_main_large_bg = 0.8
        self.alpha_main_large_gui = 0.8
        self.alpha_each_bg = 0.7
        self.alpha_each_gui = 0.8
        self.alpha_each_click_bg = 0.7
        self.alpha_each_click_gui = 0.7

        self.mode = 'None'

        self.uterus_interface_mode = 'high'   # 'high', 'wide', 'slide'
        self.load_panel_icons(self.main_panel_img_W, self.main_panel_img_H, scale=1.0)
        self.load_1080p_icons(self.main_3d_img_W, self.main_3d_img_H, scale=1.0)
        self.load_template_1080p()

        # calculation rotation angle for the initial setting
        self.LR_angle = 0
        self.UD_angle = 0

        # ---------- uterus setting --------------
        self.uterus_tip_usage = 0
        self.uterus_speed = 10  # 0.1-2F
        self.udp_flag = True    # True False   # todo
        self.udp_flag_receive = True    # True Faflse

        if self.udp_flag:
            HOST = '192.168.10.1'
            PORT = 8000
            self.addr = (HOST, PORT)
            self.UDPSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.UDPSock.bind((HOST, PORT))
            # print('...Waiting for message...')
            # self.UDOSocket_data, self.UDOSocket_client_add = self.UDPSock.recvfrom(1024)
            # print('...recieve message !!!')
        # ---------- uterus udp setting --------------

    def cameraL_job(self):
        # _, self.img_right_src_1080p = self.cap_L.read()
        # global img_LR_src
        while not rospy.is_shutdown():
            ret_L, image_L = self.cap_L.read()
            if image_L is not None:
                # Limg_ROI = image_L[3:1080, 258:1661]
                # Limg_ROI = image_L[0:1080, 95:1825]
                Limg_ROI = image_L[4:1080, 258:1662] # curi
                # Limg_ROI = image_L[0:1080, 96:1824] # cadaver2
                img_right_src_1080p_ = cv2.resize(Limg_ROI, (self.main_3d_img_W, self.main_3d_img_H))
                img_right_src_1080p = img_right_src_1080p_[self.shift_display_y:self.main_3d_img_H-self.shift_display_y, self.shift_display_x:self.main_3d_img_W-self.shift_display_x]
                self.img_right_src_1080p = cv2.resize(img_right_src_1080p, (self.main_3d_img_W, self.main_3d_img_H))

    def keyboard_job(self):
        while not rospy.is_shutdown():
            keyboard_event = keyboard.read_event() # must put in thread, otherwise the main program will keep wait the event
            if keyboard_event.event_type == keyboard.KEY_UP and keyboard_event.name == 'up':
                self.Zoomin_FIX_Mode = True
                # self.vis_setting_depth_time = time.time()
            elif keyboard_event.event_type == keyboard.KEY_UP and keyboard_event.name == 'left': # 'down'
                self.Zoomout_FIX_Mode = True
                # self.vis_setting_depth_time = time.time()
            # elif keyboard_event.event_type == keyboard.KEY_UP and keyboard_event.name == 'down': #'left'
            #     self.target_status = False  # necessary: to be used to back to initial position
            #     self.Return_Track_Mode = True
            # elif keyboard_event.event_type == keyboard.KEY_UP and keyboard_event.name == 'right':
            #     self.gui_display = True
            #     self.mode = 'Dominant'
            #     self.dominant_tool_time = time.time()
            #     self.dominant_tool_time_delay = 0
            #     self.dominant_time = time.time()
            #     self.diminant_time_delay = 0
            else:
                pass

    def white_referBBoxCallback(self, Bbox):
        self.white_bbox_refer_src = Bbox

    def yellow_referBBoxCallback(self, Bbox):
        self.yellow_bbox_refer_src = Bbox

    def visualize_whetherBBoxCallback(self, visualize):
        self.whether_visualize = visualize.data

    def joyCallback(self, Joy):
        self.joy_src = Joy

    def jointStateCallback(self, jointState):
        self.joint_state = jointState
        self.joint_pos = np.array(self.joint_state.position, dtype=float)

    def predActionCallback(self, vector3):
        self.pred_action = vector3

    def tool0_predPositionCallback(self, vector3):
        self.pred_position_tools[0] = vector3

    def tool1_predPositionCallback(self, vector3):
        self.pred_position_tools[1] = vector3

    def tool2_predPositionCallback(self, vector3):
        self.pred_position_tools[2] = vector3

    def tool3_predPositionCallback(self, vector3):
        self.pred_position_tools[3] = vector3

    def tool4_predPositionCallback(self, vector3):
        self.pred_position_tools[4] = vector3

    def toolDepthCallback(self, depth):
        self.tool_depth = depth.data

    @debug_class('MainWindow')
    def show_img(self):
        """
        show the numpy format image with the QLabel datatype
        :param img:
        :return:
        """
        # self.img_right_src_RGB = cv2.resize(self.img_right_src_RGB_1080p, (self.main_panel_img_W, self.main_panel_img_H))
        show_img = QImage(self.img_right_src_RGB.data, self.img_right_src_RGB.shape[1], self.img_right_src_RGB.shape[0],
                          QImage.Format_RGB888)
        self.label_panel_img_show.setPixmap(QPixmap.fromImage(show_img))

    def show_main_tool(self):
        main_tool_img = cv2.resize(self.main_domain_tool_ls[self.main_tool_id], (120, 80))
        main_tool_img_RGB = cv2.cvtColor(main_tool_img, cv2.COLOR_RGBA2RGB)
        show_img = QImage(main_tool_img_RGB.data, main_tool_img_RGB.shape[1], main_tool_img_RGB.shape[0],
                          QImage.Format_RGB888)
        self.label_main_img_show.setPixmap(QPixmap.fromImage(show_img))

    @debug_class('MainWindow')
    @pyqtSlot()
    def Show_L_Img_1080p_trigger(self):
        self.left_1080p = Ui_Left1080p()
        self.left_1080p.setupUi(self.left_1080p)
        self.left_1080p.show()
        self.show_L_Img_1080p_flag = True

    def Show_L_Img_1080p(self):
        if self.show_L_Img_1080p_flag is True:
            show_img = QImage(self.img_left_src_RGB_1080p.data, self.img_left_src_RGB_1080p.shape[1], self.img_left_src_RGB_1080p.shape[0],
                              QImage.Format_RGB888)
            self.left_1080p.label_imgL_1080p_show.setPixmap(QPixmap.fromImage(show_img))

    @debug_class('MainWindow')
    @pyqtSlot()
    def Show_R_Img_1080p_trigger(self):
        self.right_1080p = Ui_Right1080p()
        self.right_1080p.setupUi(self.right_1080p)
        self.right_1080p.show()
        self.show_R_Img_1080p_flag = True

    def Show_R_Img_1080p(self):
        if self.show_R_Img_1080p_flag is True:
            img_right_src_RGB_1080p_ = cv2.resize(self.img_right_src_RGB_1080p, (self.img_right_src_RGB_1080p.shape[1], self.img_right_src_RGB_1080p.shape[0] - 27))
            # show_img = QImage(self.img_right_src_RGB_1080p.data, self.img_right_src_RGB_1080p.shape[1], self.img_right_src_RGB_1080p.shape[0],
            #               QImage.Format_RGB888)
            show_img = QImage(img_right_src_RGB_1080p_.data, img_right_src_RGB_1080p_.shape[1], img_right_src_RGB_1080p_.shape[0], QImage.Format_RGB888)
            self.right_1080p.label_imgR_1080p_show.setPixmap(QPixmap.fromImage(show_img))

    @debug_class('MainWindow')
    def Turn_On_Off_system(self):
        if self.switchbutton_system.state is True:
            self.pushButton_start_clicked()
        else:
            self.pushButton_end_clicked()

    @debug_class('MainWindow')
    @pyqtSlot()
    def pushButton_start_clicked(self):
        """
        open camera if the button was clicked
        :return:
        """
        if self.img_right_src_1080p is not None:
            # Disable this button
            # self.pushButton_start.setEnabled(False)

            # acquire the image from the camera
            self.video_flag = True

            # speed_level setting
            self.image_xy_speed_coef = 1.0
            self.depth_zz_speed_coef = 1.0

            # Manual/Automatic Mode
            Manual_Mode = True
            Mode_flag = True
            # Predefined action
            self.Return_Mode = False
            self.Zoomin_Mode = False
            self.Zoomin_FIX_Mode = False
            self.Zoomout_FIX_Mode = False
            self.virtual_button_flag = False
            self.zoomin_action_once_flag = True  # used for remember the previous position
            self.Zoomout_Mode = False
            self.zoomout_action_once_flag = True  # used for remember the previous position
            self.Track_xy_Mode = False
            self.Track_z_Mode = False
            self.Emergency_Mode = False
            self.Emergency_Mode_pre = False # used for LED status use
            Emergency_flag = True
            self.Setting_Mode = False
            self.Setting_flag = True
            self.Return_Track_Mode = False
            self.Return_Track_Finish_flag = False
            # status for the predefined action: False means the target not finished; True means that the target finished
            self.target_status = True
            # flag for control the depth using predefined interval or no depth control at this moment
            self.user_depth_flag = False

            # mode of 0:pure_manual/ 1:pure_uterus/ 2:one_surgeon_four_arm
            self.Version_4_flag = False
            self.Version_mode = 0    # todo show the version_mode on the panel
            Version_flag = True

            # surgical phase
            self.Phase_Num = 4
            self.Phase_Cur = 0
            self.BoxValue_dict = {'0': 1.0, '1': 1.0, '2': 0.8, '3': 1.2}
            self.CspeedValue_dict = {'0': 1.0, '1': 1.6, '2': 2.0, '3': 1.2}
            Phase_flag = True

            # setting panel trigger by tool
            self.mode_trigger_tool = False

            # setting the initial depth workspace
            InitWorkspaceSet_flag = True

            # ------------- uterus manipulator setting ---------------------
            self.UterusUp_Mode = False
            self.UterusDown_Mode = False
            self.UterusLeft_Mode = False
            self.UterusRight_Mode = False
            self.UterusInsert_Mode = False
            self.UterusRetract_Mode = False

            # ------------- setting panel setting ---------------------
            self.SettingLap = False
            self.SettingLap0 = True
            self.SettingHand = False
            self.SettingGlobal = False
            self.SettingHandLeft = True
            self.SettingUspeed = False
            self.SettingCspeed = False
            self.SettingBox = False
            self.SettingDepth = False
            self.SettingRotation = False

            self.SettingTipUsage = False
            self.SettingTipUsage0 = 'True'   # self.uterus_tip_usage = 0, use tip
            self.SettingTipUsage_flag = True

            self.SettingGlobal0 = 'False'

            self.SettingAnteversion = '-20 +40 deg'
            self.SettingLateral = '+-30 deg'
            self.SettingInsertion = '0-50 mm'

            self.SettingLap0_flag = True
            self.SettingHand_flag = True

            self.SettingUspeedValue = 0.8
            self.SettingCspeedValue = 1.0
            self.SettingUspeedCo = 1.0
            self.ControlUspeedValue = 0.8

            # self.SettingBoxValue = round(self.bbox_scale * 50)
            self.SettingBoxValue = self.bbox_scale
            self.SettingDepthRate = 1.0
            self.SettingRotationValue = self.rotation
            self.SettingRotationPreValue = self.SettingRotationValue

            self.uterus_angle = [0, 0, 0, 0, 0, 0]  # pitch, yaw
            self.uterus_angle_control = [0, 0]  # pitch, yaw
            self.uterus_angle_pitch = 0
            self.uterus_angle_yaw = 0
            self.uterus_angle_rotation = 0
            self.uterus_angle_insertion = 0
            self.uterus_angle_grasp = 0

            uterus_angle_flag = True
            self.uterus_angle_filter = [0, 0, 0, 0, 0]
            self.uterus_angle_all = [0,0,0,0,0, 0,0,0,0,0,0]

            self.related_pitch = -3.0
            self.related_yaw = 1.0
            self.pitch_thre = 3
            self.yaw_thre = 3

            self.UterusAuto_Mode = False
            self.uterus_target_angle = [0.0, 0.0]
            self.pitch_error = 0.0
            self.yaw_error = 0.0

            # left-right view flag
            self.gui_display = False
            self.gui_display_pre = False # to detect the status change, for LED use
            self.gui_side_left = True  # == SettingHandLeft

            # action control
            self.is_out_rect = False

            # Full list
            self.tool_index = list(range(self.NUM_TOOLS_TO_TRACK))

            uterus_actions_img = [0, 0, 0]
            self.user_depth_allow_zoomout_flag = True
            self.user_depth_allow_zoomin_flag = True

            while self.video_flag:
                time_start = time.time()
                self.img_right_src_RGB_1080p = cv2.cvtColor(self.img_right_src_1080p, cv2.COLOR_BGR2RGB)
                time_cvt = time.time()
                # main tool
                self.pred_position_main = self.pred_position_tools[self.main_tool_id]
                # use the main tool information to calculate the self.pred_action
                # self.pred_action()
                self.aux_tool_index = list(range(self.NUM_TOOLS_TO_TRACK))
                del (self.aux_tool_index[self.main_tool_id])
                self.pred_position_aux = [self.pred_position_tools[i] for i in self.aux_tool_index]
                time_cvt_1 = time.time()

                self.image_xy_speed_coef = self.SettingCspeedValue
                self.depth_zz_speed_coef = self.SettingCspeedValue

                # version mode: 0: all manual mode, no gui/ 1: pure uterus / 2: osfa
                if self.joy_src.buttons[3] == 1 and Version_flag is True:
                    if self.Version_4_flag:
                        if self.Version_mode == 0:
                            self.Version_mode = 1
                        elif self.Version_mode == 1:
                            self.Version_mode = 2
                        elif self.Version_mode == 2:
                            self.Version_mode = 3
                        elif self.Version_mode == 3:
                            self.Version_mode = 0
                        Version_flag = False
                    else:
                        if self.Version_mode == 0:
                            self.Version_mode = 1
                        elif self.Version_mode == 1:
                            self.Version_mode = 2
                        elif self.Version_mode == 2:
                            self.Version_mode = 0
                        Version_flag = False
                if self.joy_src.buttons[3] == 0 and Version_flag is False:
                    Version_flag = True
                print('Version_Mode', self.Version_mode, 'Version_flag', Version_flag)
                # cv2.putText(self.img_right_src_1080p, str(self.Version_mode), (100, 100), 0, 2, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                # adjust the surgical phase: the press 3 is only valid when the system is not in Setting mode
                if (not self.Setting_Mode) and self.mode != 'Setting':
                    if self.joy_src.buttons[2] == 1 and Phase_flag is True:
                        self.Phase_Cur = (self.Phase_Cur + 1) % self.Phase_Num
                        # Setting box value
                        self.SettingBoxValue = self.BoxValue_dict[str(int(self.Phase_Cur))]
                        self.SettingCspeedValue = self.CspeedValue_dict[str(int(self.Phase_Cur))]
                        Phase_flag = False
                    if self.joy_src.buttons[2] == 0 and Phase_flag is False:
                        Phase_flag = True

                # adjust the left-right panel
                if self.Version_mode == 2 or self.Version_mode == 3:
                    if self.joy_src.buttons[0] == 1 and Mode_flag is True:
                        Manual_Mode = not Manual_Mode  # switch to other mode
                        Mode_flag = False
                    if self.joy_src.buttons[0] == 0 and Mode_flag is False:
                        Mode_flag = True
                else:
                    Manual_Mode = True
                print('Manual_Mode', Manual_Mode, 'Mode_flag', Mode_flag)

                # flag change the white and yellow box
                if self.joy_src.buttons[1] == 1 and self.Setting_flag is True:
                    self.Setting_Mode = not self.Setting_Mode
                    self.Setting_flag = False
                if self.joy_src.buttons[1] == 0 and self.Setting_flag is False:
                    self.Setting_flag = True

                if self.joy_src.buttons[4] == 1 and Emergency_flag is True:
                    self.Emergency_Mode = not self.Emergency_Mode
                    Emergency_flag = False
                if self.joy_src.buttons[4] == 0 and Emergency_flag is False:
                    Emergency_flag = True

                # flag zoom in/out with fixed button
                # print('keep wait for keypressing ...')

                time_draw_1 = time.time()
                # GUI button triggered semi-automation subtasks
                if self.Emergency_Mode is True:
                    self.Return_Track_Mode = self.Return_Mode = self.Zoomin_Mode = self.Zoomout_Mode = self.Track_xy_Mode = self.Track_z_Mode = False
                    Manual_Mode = True
                    self.target_status = True
                    self.user_depth_flag = False
                elif Manual_Mode is True:  # LT button
                    # Using the network to detect the image
                    # TODO
                    if self.Setting_Mode is True or self.mode == 'Setting':
                        self.gui_display = True
                        self.pushButton_Setup()
                        self.robot_correct_misorientation()
                        # fixed for Bug: press button once to hidden the panel
                        if self.Setting_flag is False and self.mode_trigger_tool:
                            self.mode_trigger_tool = False
                            self.Setting_Mode = False
                            self.mode = 'None'
                            self.gui_display = False
                            self.whether_visualize_in_setting = False

                        if self.Setting_flag is False and self.Setting_Mode is False:  # Setting_mode is False: the 2 is not pressed before
                            self.mode = 'None'
                            self.gui_display = False
                            self.whether_visualize_in_setting = False
                    elif self.mode == 'Uterus' or self.mode == 'Dominant':
                        # ADD in 20220409: avoid moving laparoscope in Uterus mode
                        # No operation needs to be done in manual mode
                        self.actions = [0.0, 0.0, 0.0]
                        pass
                    else:
                        self.SettingRotationPreValue = self.SettingRotationValue
                        # self.gui_side_left = self.SettingHandLeft
                        # # drive the ur5 robot
                        time_manual = time.time()
                        self.pushButton_Test_clicked()  # the button in joystick cannot control robot and setting simultaniouly
                        time_manual_1 = time.time()
                        print('time_manual_1', time_manual_1 - time_manual)
                    actions_img = [-self.actions[0] / self.image_xy_speed_coef,
                                   -self.actions[1] / self.image_xy_speed_coef,
                                   self.actions[2] / self.depth_zz_speed_coef]  # may need to multiply -1 in actions[0 or 1]
                    print('self.mode: ', self.mode)
                elif self.Return_Mode is True:
                    self.move_init()
                    actions_img = [0, 0, 0]  # may need to multiply -1 in actions[0 or 1]
                elif self.Zoomin_Mode is True:
                    self.zoom_in()
                    actions_img = [0, 0, 0]  # may need to multiply -1 in actions[0 or 1]
                    self.user_depth_flag = False #True  # No automatic depth control this time
                    self.Return_Track_Mode = False
                    self.Return_Track_Finish_flag = False
                    # print('Zoomin mode')
                elif self.Zoomout_Mode is True:
                    self.zoom_out()
                    actions_img = [0, 0, 0]  # may need to multiply -1 in actions[0 or 1]
                    self.user_depth_flag = False #True  # No automatic depth control this time
                    self.Return_Track_Mode = False
                    self.Return_Track_Finish_flag = False
                    # print('Zoomout mode')
                elif self.Track_xy_Mode is True:
                    self.semi_track_xy()
                    actions_img = [-self.actions[0] / self.image_xy_speed_coef,
                                   -self.actions[1] / self.image_xy_speed_coef,
                                   0.0]  # may need to multiply -1 in actions[0 or 1]
                elif self.Track_z_Mode is True:
                    self.Return_Track_Mode = False  # init: always track the tool in xy axis; click the track z will zoom in/out in predefined, once finished, back to manual/automatic mode
                    self.user_depth_flag = False  # use automatic depth control this time
                    self.semi_track_depth()
                    actions_img = [0.0, 0.0, self.actions[2] / self.depth_zz_speed_coef]  # may need to multiply -1 in actions[0 or 1]
                elif self.Return_Track_Mode is True:
                    # support setting in the long-time global action
                    if self.Setting_Mode is True or self.mode == 'Setting':
                        self.gui_display = True
                        self.pushButton_Setup()
                        self.robot_correct_misorientation()
                        # fixed for Bug: press button once to hidden the panel
                        if self.Setting_flag is False and self.mode_trigger_tool:
                            self.mode_trigger_tool = False
                            self.Setting_Mode = False
                            self.mode = 'None'
                            self.gui_display = False
                            self.whether_visualize_in_setting = False

                        if self.Setting_flag is False and self.Setting_Mode is False:  # Setting_mode is False: the 2 is not pressed before
                            self.mode = 'None'
                            self.gui_display = False
                            self.whether_visualize_in_setting = False
                    if self.target_status is False:
                        # Here we double the retract speed for high efficiency
                        # self.SettingCspeedValue = 2.0
                        self.Return_Track_Finish_flag = False
                        # first step
                        self.move_init()
                        actions_img = [0, 0, 0]  # may need to multiply -1 in actions[0 or 1]
                    else:
                        # second step
                        # Here we re-back to the orginal speed
                        # self.SettingCspeedValue = 1.0
                        self.Return_Track_Finish_flag = True
                        if self.mode == 'Uterus' or self.mode == 'Dominant':
                            # ADD in 20220714: avoid moving laparoscope in Uterus mode (in GLOBAL-HOLD state)
                            # No operation needs to be done here: avoid run automatic_ctrl() function
                            self.actions = [0.0, 0.0, 0.0]  # only for visualize
                        else:
                            self.semi_track_xy()
                        actions_img = [-self.actions[0] / self.image_xy_speed_coef,
                                       -self.actions[1] / self.image_xy_speed_coef,
                                       0.0]  # may need to multiply -1 in actions[0 or 1]

                else:
                    if Manual_Mode is False:
                        if self.Setting_Mode is True or self.mode == 'Setting':
                            self.gui_display = True
                            self.pushButton_Setup()
                            self.robot_correct_misorientation()
                            # fixed for Bug: press button once to hidden the panel
                            if self.Setting_flag is False and self.mode_trigger_tool:
                                self.mode_trigger_tool = False
                                self.Setting_Mode = False
                                self.mode = 'None'
                                self.gui_display = False
                                self.whether_visualize_in_setting = False

                            if self.Setting_flag is False and self.Setting_Mode is False:  # Setting_mode is False: the 2 is not pressed before
                                self.mode = 'None'
                                self.gui_display = False
                                self.whether_visualize_in_setting = False
                        elif self.mode == 'Uterus' or self.mode == 'Dominant':
                            # ADD in 20220409: avoid moving laparoscope in Uterus mode
                            # No operation needs to be done here: avoid run automatic_ctrl() function
                            self.actions = [0.0, 0.0, 0.0]  # only for visualize
                            pass
                        else:
                            self.SettingRotationPreValue = self.SettingRotationValue
                            # pass
                            # self.gui_side_left = self.SettingHandLeft # TODO: set the gui side automatically
                            self.automatic_ctrl()  # the automatic mode and modify setting can be done simultaniouly->damping in correct misorientation, don't allow control simultaniouly

                        actions_img = [-self.actions[0] / self.image_xy_speed_coef,
                                       -self.actions[1] / self.image_xy_speed_coef,
                                       self.actions[2] / self.depth_zz_speed_coef]  # may need to multiply -1 in actions[0 or 1]
                        print('Automatic mode')
                print('need to attach the joystick')
                time_cvt_3 = time.time()

                print('time cost 1: {}/{}/{}'.format(time_cvt_1 - time_cvt, time_draw_1 - time_cvt_1,
                                                     time_cvt_3 - time_draw_1))

                ######################## these operations are only calculated when zoom in/out is activated
                # monitor the length between the camera and the rcm position

                bTc = self.robot.get_bTc(self.joint_pos)
                self.dist_r2c = np.linalg.norm(bTc[0:3, 3] - self.robot.bPr_init) * 1000 # mm
                # print('self.dist_r2c: ', self.dist_r2c)

                if self.dist_r2c < 10: # avoid the camera is out of aban in automatic mode
                    self.user_depth_allow_zoomout_flag = False
                else:
                    self.user_depth_allow_zoomout_flag = True

                if self.Workspace_Inter > 0:
                    if self.Workspace_Inter - self.dist_r2c < self.closest_dist: # avoid the camera is out of aban in automatic mode
                        self.user_depth_allow_zoomin_flag = False
                    else:
                        self.user_depth_allow_zoomin_flag = True

                # if self.joy_src.buttons[5] == 1 and InitWorkspaceSet_flag is True:
                #     pred_action = np.array([self.pred_action.x, self.pred_action.y, self.pred_action.z])
                #     if self.tool_depth * 1000 > -1 and np.linalg.norm(pred_action) < 0.15 and Manual_Mode is not True:
                #         self.Workspace_Inter = self.tool_depth * 1000 + self.dist_r2c # 75 is the preset depth interval: self.mid_depth, mm
                #         InitWorkspaceSet_flag = False
                #
                #         self.mid_depth = np.clip(self.tool_depth * 1000, 30, 110)
                #         self.mid_range = 2/7 * self.mid_depth - 10 / 7
                # if self.joy_src.buttons[5] == 0 and InitWorkspaceSet_flag is False:
                #     InitWorkspaceSet_flag = True
                # print('InitWorkspaceSet flag: ', InitWorkspaceSet_flag)
                pred_action = np.array([self.pred_action.x, self.pred_action.y, 0.0])
                if self.tool_depth * 1000 > -1 and np.linalg.norm(pred_action) < 0.5 and Manual_Mode is not True:
                    System_initDepth_flag = True
                    if self.joy_src.buttons[5] == 1 and InitWorkspaceSet_flag is True:
                        self.Workspace_Inter = self.tool_depth * 1000 + self.dist_r2c # 75 is the preset depth interval: self.mid_depth, mm
                        InitWorkspaceSet_flag = False

                        # self.mid_depth = np.clip(self.tool_depth * 1000, 30, 110)
                        # self.mid_depth = self.tool_depth * 1000
                        # self.mid_depth = self.mid_depth_init
                        # self.mid_range = (2/7 * self.mid_depth - 10 / 7) * self.SettingDepthRate
                    if self.joy_src.buttons[5] == 0 and InitWorkspaceSet_flag is False:
                        InitWorkspaceSet_flag = True
                else:
                    System_initDepth_flag = False
                print('InitWorkspaceSet flag: ', InitWorkspaceSet_flag)

                # calculate the range according to the current depth
                ##################################### current manner
                # if self.Zoomin_Mode is True or self.Zoomout_Mode is True:
                #     if self.Workspace_Inter > 0: # is valid
                #         self.mid_depth = np.clip(self.mid_depth_refer - self.dist_r2c + self.dist_r2c_refer, self.closest_dist, 110)
                #     else:
                #         self.mid_depth = self.mid_depth_init
                # else:
                #     self.dist_r2c_refer = self.dist_r2c
                #     self.mid_depth_refer = self.mid_depth
                #
                # if self.Return_Track_Mode is True:
                #     self.mid_depth = self.Workspace_Inter - self.dist_r2c
                #
                # self.mid_range = (2/7 * self.mid_depth - 10 / 7) * self.SettingDepthRate
                ########################################## modified button fixed depth control
                # button triggered fixed depth+range settings
                if self.Return_Track_Mode is False: # back into initial position
                    if self.Zoomin_FIX_Mode is True:
                        # self.SettingCspeedValue = 2.0
                        self.mid_idx -= 1
                        self.Zoomin_FIX_Mode = False
                        self.Return_Track_Mode = False # disable the trackxy --> enable trackxyz
                    if self.Zoomout_FIX_Mode is True:
                        # self.SettingCspeedValue = 2.0
                        self.mid_idx += 1
                        self.Zoomout_FIX_Mode = False
                        self.Return_Track_Mode = False # disable the trackxy --> enable trackxyz
                else:
                    # Current Return track mode (# avoid global press zoom out has zoom in effect.)
                    if self.Zoomin_FIX_Mode is True: # press zoomin --> with the mid settings; ignore zoomout-press action
                        # self.SettingCspeedValue = 2.0
                        self.mid_idx = 1 # zoom in effect under the proper mid-depth and range (here we choose 1)
                        self.Zoomin_FIX_Mode = False
                        self.Return_Track_Mode = False # disable the trackxy --> enable trackxyz
                    # NOTE: the below code is used to clear the zoomout press action !!!
                    # otherwise it will be keep true and used for next iteration
                    self.Zoomout_FIX_Mode = False

                self.mid_idx = np.clip(self.mid_idx, 0, len(self.mid_depth_fix)-1)
                self.mid_depth = self.mid_depth_fix[self.mid_idx]
                self.mid_range = self.mid_range_fix[self.mid_idx]

                # calculate the two angle for controlling uterus manipulator
                self.laparoscope_z_axis_init = self.robot.bPc_init - self.robot.bPr_init
                base_z_axis = np.array([0, 0, 1])
                normal_plane1 = np.cross(base_z_axis, self.laparoscope_z_axis_init)
                normal_plane1 = normal_plane1 / (np.linalg.norm(normal_plane1) + 1e-6)
                normal_plane2 = np.cross(self.laparoscope_z_axis_init, normal_plane1)
                normal_plane2 = normal_plane2 / (np.linalg.norm(normal_plane2) + 1e-6)

                bTc_cur = self.robot.get_bTc(self.joint_pos)
                self.laparoscope_z_axis_cur = bTc_cur[0:3, 3] - self.robot.bPr_init
                dist1 = np.dot(self.laparoscope_z_axis_cur, normal_plane1) / (np.linalg.norm(normal_plane1) + 1e-6)
                dist2 = np.dot(self.laparoscope_z_axis_cur, normal_plane2) / (np.linalg.norm(normal_plane2) + 1e-6)

                proj1 = self.laparoscope_z_axis_cur - dist1 * normal_plane1
                proj2 = self.laparoscope_z_axis_cur - dist2 * normal_plane2

                self.UD_angle = np.arccos(np.clip((np.dot(proj1, self.laparoscope_z_axis_init) / (np.linalg.norm(proj1) * np.linalg.norm(self.laparoscope_z_axis_init)) + 1e-6), -1, 1)) * 180/np.pi
                self.LR_angle = np.arccos(np.clip((np.dot(proj2, self.laparoscope_z_axis_init) / (np.linalg.norm(proj2) * np.linalg.norm(self.laparoscope_z_axis_init)) + 1e-6), -1, 1)) * 180/np.pi

                UD_angle_sign = 1 if np.dot(np.cross(proj1, self.laparoscope_z_axis_init), normal_plane1) > 0 else -1
                LR_angle_sign = 1 if np.dot(np.cross(proj2, self.laparoscope_z_axis_init), normal_plane2) > 0 else -1

                self.UD_angle = self.UD_angle * UD_angle_sign
                self.LR_angle = self.LR_angle * LR_angle_sign

                # ---------- uterus action ---------------
                self.UterusAuto_Mode = False
                self.action_uterus = [0, 0, 0, 0, 0, 0]  # up, down, left, right, insert, retract
                if self.Version_mode == 1 or self.Version_mode == 2:
                    if self.UterusUp_Mode is True:  # y  # todo emergency mode of the uterus manipulator
                        self.action_uterus = [1, 0, 0, 0, 0, 0]
                        uterus_actions_img = [0, 1, 0]  # may need to multiply -1 in actions[0 or 1]
                        # print('UterusUp mode')
                        self.UterusUp_Mode = False
                    elif self.UterusDown_Mode is True:
                        self.action_uterus = [0, 1, 0, 0, 0, 0]
                        uterus_actions_img = [0, -1, 0]  # may need to multiply -1 in actions[0 or 1]
                        # print('UterusDown mode')
                        self.UterusDown_Mode = False
                    elif self.UterusLeft_Mode is True:  # x
                        self.action_uterus = [0, 0, 1, 0, 0, 0]
                        uterus_actions_img = [1, 0, 0]  # may need to multiply -1 in actions[0 or 1]
                        # print('UterusLeft mode')
                        self.UterusLeft_Mode = False
                    elif self.UterusRight_Mode is True:
                        self.action_uterus = [0, 0, 0, 1, 0, 0]
                        uterus_actions_img = [-1, 0, 0]  # may need to multiply -1 in actions[0 or 1]
                        # print('UterusRight mode')
                        self.UterusRight_Mode = False
                    elif self.UterusInsert_Mode is True:  # z
                        self.action_uterus = [0, 0, 0, 0, 1, 0]
                        uterus_actions_img = [0, 0, 1]  # may need to multiply -1 in actions[0 or 1]
                        # print('UterusInsert mode')
                        self.UterusInsert_Mode = False
                    elif self.UterusRetract_Mode is True:
                        self.action_uterus = [0, 0, 0, 0, 0, 1]
                        uterus_actions_img = [0, 0, -1]  # may need to multiply -1 in actions[0 or 1]
                        # print('UterusRetract mode')
                        self.UterusRetract_Mode = False
                    else:
                        uterus_actions_img = [0, 0, 0]

                elif self.Version_mode == 3:   # todo two-robot motion
                    self.uterus_target_angle = [self.related_pitch * self.UD_angle, self.related_yaw * self.LR_angle]

                    self.pitch_error = self.uterus_target_angle[0] - self.uterus_angle_control[0]
                    self.yaw_error = self.uterus_target_angle[1] - self.uterus_angle_control[1]

                    if abs(self.pitch_error) <= self.pitch_thre:
                        self.action_uterus[0] = 0
                        self.action_uterus[1] = 0
                    elif self.pitch_error > 0:
                        self.SettingUspeedValue = 1.0
                        self.UterusAuto_Mode = True
                        self.action_uterus[0] = 1
                        self.action_uterus[1] = 0
                    elif self.pitch_error < 0:
                        self.SettingUspeedValue = 1.0
                        self.UterusAuto_Mode = True
                        self.action_uterus[0] = 0
                        self.action_uterus[1] = 1

                    if abs(self.pitch_error) <= self.pitch_thre:
                        self.SettingUspeedValue = 0.5
                        if abs(self.yaw_error) <= self.yaw_thre:
                            self.action_uterus[2] = 0
                            self.action_uterus[3] = 0
                        elif self.yaw_error > 0:
                            self.UterusAuto_Mode = True
                            self.action_uterus[2] = 1
                            self.action_uterus[3] = 0
                        elif self.yaw_error < 0:
                            self.UterusAuto_Mode = True
                            self.action_uterus[2] = 0
                            self.action_uterus[3] = 1

                    if self.uterus_angle_pitch >= 60:
                        self.action_uterus[0] = 0

                    if self.uterus_angle_pitch <= -15:
                        self.action_uterus[1] = 0

                    if self.uterus_angle_yaw >= 20:
                        self.action_uterus[2] = 0

                    if abs(self.uterus_angle_yaw) <= -20:
                        self.action_uterus[3] = 0

                    self.action_uterus[4] = 0
                    self.action_uterus[5] = 0
                    print(self.action_uterus)

                if self.SettingTipUsage0 == 'True':
                    self.uterus_tip_usage = 0
                else:
                    self.uterus_tip_usage = 1

                if self.udp_flag:
                    # self.uterus_speed = int(self.SettingUspeedValue*10)
                    self.uterus_speed = int(self.ControlUspeedValue*10)
                    action_uterus_data = bytearray(
                        [self.action_uterus[0], self.action_uterus[1], self.action_uterus[3], self.action_uterus[2],
                         self.action_uterus[4], self.action_uterus[5], self.uterus_tip_usage, self.uterus_speed])

                    # with open('udp_save.txt', 'r') as udp_save_file:
                    #     udp_port = udp_save_file.readline()[-6:-1]
                    # self.UDPSock.sendto(action_uterus_data, ('192.168.10.10', int(udp_port)))
                    self.UDPSock.sendto(action_uterus_data, ('192.168.10.10', 8001))
                    print('action_uterus_data', action_uterus_data)
                    # self.UDPSock.sendto(action_uterus_data, self.UDOSocket_client_add)

                if self.udp_flag_receive:
                    # yaw (L229/-27~R27), pitch (D233/-21~U40), insertion (1~50)
                    # tool tilt (1~26), tool grasp (1~46/47)
                    UDOSocket_data = self.UDPSock.recvfrom(1024)[0]
                    print('-------- UDOSocket_data -------', UDOSocket_data)

                    if UDOSocket_data[1] > 200:
                        uterus_angle_pitch_robot = UDOSocket_data[1] - 256
                        self.uterus_angle_pitch = UDOSocket_data[1] - 256 + UDOSocket_data[3]
                    else:
                        uterus_angle_pitch_robot = UDOSocket_data[1]
                        self.uterus_angle_pitch = UDOSocket_data[1] + UDOSocket_data[3]

                    if UDOSocket_data[0] > 100:
                        self.uterus_angle_yaw = UDOSocket_data[0] - 256
                    else:
                        self.uterus_angle_yaw = UDOSocket_data[0]

                    if UDOSocket_data[3] > 100:
                        self.uterus_angle_rotation = UDOSocket_data[3] - 256
                    else:
                        self.uterus_angle_rotation = UDOSocket_data[3]

                    if UDOSocket_data[2] > 250:
                        self.uterus_angle_insertion = UDOSocket_data[2] - 256
                    else:
                        self.uterus_angle_insertion = UDOSocket_data[2]
                    # self.uterus_angle_insertion = UDOSocket_data[2]

                    self.uterus_angle_grasp = UDOSocket_data[5]

                    self.uterus_angle = [self.uterus_angle_yaw, uterus_angle_pitch_robot, self.uterus_angle_insertion,
                                         UDOSocket_data[3], self.uterus_angle_rotation, self.uterus_angle_grasp]

                    if uterus_angle_flag:
                        uterus_angle_flag = False
                        uterus_angle_pitch_pre = self.uterus_angle_pitch
                        uterus_angle_yaw_pre = self.uterus_angle_yaw
                        uterus_angle_rotation_pre = self.uterus_angle_rotation
                        uterus_angle_insertion_pre = self.uterus_angle_insertion
                        uterus_angle_grasp_pre = self.uterus_angle_grasp

                    if self.uterus_angle_pitch > 68 or self.uterus_angle_pitch < -23 or \
                            abs(self.uterus_angle_pitch - uterus_angle_pitch_pre) >= 6:
                        self.uterus_angle_pitch = uterus_angle_pitch_pre
                    else:
                        uterus_angle_pitch_pre = self.uterus_angle_pitch

                    if self.uterus_angle_yaw > 30 or self.uterus_angle_yaw < -30 or \
                            abs(self.uterus_angle_yaw - uterus_angle_yaw_pre) >= 6:
                        self.uterus_angle_yaw = uterus_angle_yaw_pre
                    else:
                        uterus_angle_yaw_pre = self.uterus_angle_yaw

                    if self.uterus_angle_rotation > 30 or self.uterus_angle_rotation < 0 or \
                            abs(self.uterus_angle_rotation - uterus_angle_rotation_pre) >= 6:
                        self.uterus_angle_rotation = uterus_angle_rotation_pre
                    else:
                        uterus_angle_rotation_pre = self.uterus_angle_rotation

                    if self.uterus_angle_insertion > 53 or self.uterus_angle_insertion < 0 or \
                            abs(self.uterus_angle_insertion - uterus_angle_insertion_pre) >= 8:
                        self.uterus_angle_insertion = uterus_angle_insertion_pre
                    else:
                        uterus_angle_insertion_pre = self.uterus_angle_insertion

                    if self.uterus_angle_grasp > 50 or self.uterus_angle_grasp < 0 or \
                            abs(self.uterus_angle_grasp - uterus_angle_grasp_pre) >= 8:
                        self.uterus_angle_grasp = uterus_angle_grasp_pre
                    else:
                        uterus_angle_grasp_pre = self.uterus_angle_grasp

                    self.uterus_angle_filter = [self.uterus_angle_pitch, self.uterus_angle_yaw,
                                                self.uterus_angle_rotation, self.uterus_angle_insertion,
                                                self.uterus_angle_grasp]
                    self.uterus_angle_all[:6] = self.uterus_angle
                    self.uterus_angle_all[-5:] = self.uterus_angle_filter
                    print('self.uterus_angle_filter', self.uterus_angle_filter)

                    self.uterus_angle_control = [self.uterus_angle_pitch, self.uterus_angle_yaw]

                    # # todo uterus speed control
                    if self.uterus_angle_pitch > 64 or self.uterus_angle_pitch < -19 or self.uterus_angle_yaw > 25 or\
                            self.uterus_angle_yaw < -25:
                        self.SettingUspeedCo = 0.6
                    elif self.uterus_angle_pitch > 61 or self.uterus_angle_pitch < -16 or self.uterus_angle_yaw > 23 or\
                            self.uterus_angle_yaw < -23 or self.uterus_angle_insertion > 45:
                        self.SettingUspeedCo = 0.8
                    else:
                        self.SettingUspeedCo = 1.0

                    # uspeed_value = self.SettingUspeedValue
                    # self.SettingUspeedValue = self.SettingUspeedCo * self.SettingUspeedValue
                    self.ControlUspeedValue = self.SettingUspeedCo * self.SettingUspeedValue

                print('self.uterus_angle', self.uterus_angle, self.uterus_angle_control)
                # ----------- read uterus angle -------------

                if self.Emergency_Mode is True:
                    Mode = f'Mode: Emergency'
                elif self.Return_Mode is True:
                    Mode = f'Mode: Return'
                elif self.Zoomin_Mode is True:
                    Mode = f'Mode: Zoom_in'
                elif self.Zoomout_Mode is True:
                    Mode = f'Mode: Zoom_out'
                elif self.Track_xy_Mode is True:
                    Mode = f'Mode: Track_XY'
                elif self.Track_z_Mode is True:
                    Mode = f'Mode: Track_Z'
                elif self.Return_Track_Mode is True:
                    Mode = f'Mode: ReturnTrack'
                else:
                    Mode = f'Mode: Manual' if Manual_Mode is True else f'Mode: Automatic'

                if self.mode == 'Camera':
                    gui_mode = f'Mode: Camera'
                elif self.mode == 'Uterus':
                    gui_mode = f'Mode: Uterus'
                elif self.mode == 'Setting':
                    gui_mode = f'Mode: Setting'
                elif self.mode == "Dominant":
                    gui_mode = f'Mode: Main'
                else:
                    gui_mode = f'Mode: None'

                # For debug
                # RCM2CAM_display = f'RCM2CAM Len: {self.dist_r2c}'
                # cv2.putText(self.img_right_src_RGB_1080p, RCM2CAM_display, (150, 80), 0, 0.5, [225, 255, 255], thickness=1,
                #                     lineType=cv2.LINE_AA)
                #
                # RCM2CAM_REFER_display = f'RCM2CAM Len: {self.dist_r2c_refer}'
                # cv2.putText(self.img_right_src_RGB_1080p, RCM2CAM_REFER_display, (150, 120), 0, 0.5, [225, 255, 255], thickness=1,
                #                     lineType=cv2.LINE_AA)
                #
                # MidDepth_display = f'Mid Depth: {self.mid_depth}'
                # # MidDepth_display = f'Mid Depth: {self.Workspace_Inter - self.dist_r2c}'
                # cv2.putText(self.img_right_src_RGB_1080p, MidDepth_display, (150, 150), 0, 0.5, [225, 255, 255], thickness=1,
                #                     lineType=cv2.LINE_AA)
                # DepthRange_display = f'Depth Range: {self.mid_range}'
                # cv2.putText(self.img_right_src_RGB_1080p, DepthRange_display, (150, 200), 0, 0.5, [225, 255, 255], thickness=1,
                #                     lineType=cv2.LINE_AA)

                # if self.Zoomin_FIX_Mode is True or self.Zoomout_FIX_Mode is True:
                #     if time.time() - self.vis_setting_depth_time > self.vis_setting_depth_time_delay:
                #         MidDepth_display = f'Mid Depth: {self.mid_depth}'
                #         # MidDepth_display = f'Mid Depth: {self.Workspace_Inter - self.dist_r2c}'
                #         cv2.putText(self.img_right_src_RGB_1080p, MidDepth_display, (150, 150), 0, 0.5, [225, 255, 255], thickness=1,
                #                             lineType=cv2.LINE_AA)
                #         DepthRange_display = f'Depth Range: {self.mid_range}'
                #         cv2.putText(self.img_right_src_RGB_1080p, DepthRange_display, (150, 200), 0, 0.5, [225, 255, 255], thickness=1,
                #                             lineType=cv2.LINE_AA)


                # mid_idx_display = f'MID_IDX: {self.mid_idx}'
                # cv2.putText(self.img_right_src_RGB_1080p, mid_idx_display, (150, 80), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                cVc_display = f'Vx: {self.robot.cVc_norm[0]}, Vy: {self.robot.cVc_norm[1]}, Vz: {self.robot.cVc_norm[2]}'
                # cv2.putText(self.img_right_src_RGB_1080p, cVc_display, (150, 80), 0, 0.5, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                # draw the position
                scale_x_right_1080p = self.main_3d_img_W / 640 # 3.0  # 1920 / 640
                scale_y_right_1080p = self.main_3d_img_H / 480 # 2.25  # 1080 / 480

                positions_aux_img = []
                positions_aux_img_1080p = []
                for i in range(len(self.pred_position_aux)):
                    # print('len: {}, index: {}, aux: {}'.format(len(self.pred_position_aux), i, self.pred_position_aux[i]))
                    if (not np.isnan(self.pred_position_aux[i].x)) and (not np.isnan(self.pred_position_aux[i].y)):
                        position_aux_img_1080p = [int(self.pred_position_aux[i].x * scale_x_right_1080p),
                                                  int(self.pred_position_aux[i].y * scale_y_right_1080p)]
                        # if int(position_aux_img_1080p[0]) < (self.main_3d_img_W-self.shift_display/1920*self.main_3d_img_W):
                        #     offset_scale_x_right_1080p = self.main_3d_img_W / (self.main_3d_img_W-self.shift_display/1920*self.main_3d_img_W)
                        #     offset_scale_y_right_1080p = 1
                        # else:
                        #     offset_scale_x_right_1080p = -1
                        #     offset_scale_y_right_1080p = -1
                        # position_aux_img_1080p = [int(position_aux_img_1080p[0] * offset_scale_x_right_1080p),
                        #                           int(position_aux_img_1080p[1] * offset_scale_y_right_1080p)]
                        # position_aux_img = [int(position_aux_img_1080p[0] * self.main_panel_img_W / self.main_3d_img_W),
                        #                     int(position_aux_img_1080p[1] * self.main_panel_img_H / self.main_3d_img_H)]
                        position_aux_img_1080p_pixel_coordinate = [position_aux_img_1080p[0] - self.main_3d_img_W /2,
                                                                   position_aux_img_1080p[1] - self.main_3d_img_H /2]

                        if abs(position_aux_img_1080p_pixel_coordinate[0]) < (self.main_3d_img_W*self.shift_scale/2) and \
                           abs(position_aux_img_1080p_pixel_coordinate[1]) < (self.main_3d_img_H*self.shift_scale/2):
                            shift_scale_x_right_1080p = 1/self.shift_scale
                            shift_scale_y_right_1080p = 1/self.shift_scale
                        # else:
                        #     shift_scale_x_right_1080p = -1
                        #     shift_scale_y_right_1080p = -1
                            position_aux_img_1080p_pixel_coordinate = [int(position_aux_img_1080p_pixel_coordinate[0] * shift_scale_x_right_1080p),
                                                                       int(position_aux_img_1080p_pixel_coordinate[1] * shift_scale_y_right_1080p)]
                            position_aux_img_1080p = [int(position_aux_img_1080p_pixel_coordinate[0] + self.main_3d_img_W/2),
                                                      int(position_aux_img_1080p_pixel_coordinate[1] + self.main_3d_img_H/2)]
                            position_aux_img = [int(position_aux_img_1080p[0] * self.main_panel_img_W / self.main_3d_img_W),
                                                int(position_aux_img_1080p[1] * self.main_panel_img_H / self.main_3d_img_H)]
                        elif abs(position_aux_img_1080p_pixel_coordinate[0]) < (self.main_3d_img_W*self.shift_scale_appear/2) and \
                             abs(position_aux_img_1080p_pixel_coordinate[1]) < (self.main_3d_img_H*self.shift_scale_appear/2):

                            shift_scale_x_right_1080p = 1/self.shift_scale
                            shift_scale_y_right_1080p = 1/self.shift_scale
                            position_aux_img_1080p_pixel_coordinate = [int(position_aux_img_1080p_pixel_coordinate[0] * shift_scale_x_right_1080p),
                                                                       int(position_aux_img_1080p_pixel_coordinate[1] * shift_scale_y_right_1080p)]
                            position_aux_img_1080p = [int(position_aux_img_1080p_pixel_coordinate[0] + self.main_3d_img_W/2),
                                                      int(position_aux_img_1080p_pixel_coordinate[1] + self.main_3d_img_H/2)]
                            position_aux_img_1080p = [min(max(20,position_aux_img_1080p[0]), self.main_3d_img_W-20),
                                                      min(max(20,position_aux_img_1080p[1]), self.main_3d_img_H-20)]
                            position_aux_img = [int(position_aux_img_1080p[0] * self.main_panel_img_W / self.main_3d_img_W),
                                                int(position_aux_img_1080p[1] * self.main_panel_img_H / self.main_3d_img_H)]
                        else:
                            position_aux_img = [-1, -1]
                            position_aux_img_1080p = [-1, -1]
                    else:
                        position_aux_img = [-1, -1]
                        position_aux_img_1080p = [-1, -1]
                    positions_aux_img.append(position_aux_img)
                    positions_aux_img_1080p.append(position_aux_img_1080p)

                if (not np.isnan(self.pred_position_main.x)) and (not np.isnan(self.pred_position_main.y)):
                    positions_main_img_1080p = [int(self.pred_position_main.x * scale_x_right_1080p),
                                               int(self.pred_position_main.y * scale_y_right_1080p)]
                    positions_main_img_1080p_pixel_coordinate = [positions_main_img_1080p[0] - self.main_3d_img_W /2,
                                                               positions_main_img_1080p[1] - self.main_3d_img_H /2]

                    if abs(positions_main_img_1080p_pixel_coordinate[0]) < (self.main_3d_img_W*self.shift_scale/2) and \
                       abs(positions_main_img_1080p_pixel_coordinate[1]) < (self.main_3d_img_H*self.shift_scale/2):
                        shift_scale_x_right_1080p = 1/self.shift_scale
                        shift_scale_y_right_1080p = 1/self.shift_scale

                        positions_main_img_1080p_pixel_coordinate = [int(positions_main_img_1080p_pixel_coordinate[0] * shift_scale_x_right_1080p),
                                                                   int(positions_main_img_1080p_pixel_coordinate[1] * shift_scale_y_right_1080p)]
                        positions_main_img_1080p = [int(positions_main_img_1080p_pixel_coordinate[0] + self.main_3d_img_W/2),
                                                  int(positions_main_img_1080p_pixel_coordinate[1] + self.main_3d_img_H/2)]
                        positions_main_img = [int(positions_main_img_1080p[0] * self.main_panel_img_W / self.main_3d_img_W),
                                            int(positions_main_img_1080p[1] * self.main_panel_img_H / self.main_3d_img_H)]

                    elif abs(positions_main_img_1080p_pixel_coordinate[0]) < (self.main_3d_img_W*self.shift_scale_appear/2) and \
                         abs(positions_main_img_1080p_pixel_coordinate[1]) < (self.main_3d_img_H*self.shift_scale_appear/2):

                        shift_scale_x_right_1080p = 1/self.shift_scale
                        shift_scale_y_right_1080p = 1/self.shift_scale
                        positions_main_img_1080p_pixel_coordinate = [int(positions_main_img_1080p_pixel_coordinate[0] * shift_scale_x_right_1080p),
                                                                   int(positions_main_img_1080p_pixel_coordinate[1] * shift_scale_y_right_1080p)]
                        positions_main_img_1080p_ = [int(positions_main_img_1080p_pixel_coordinate[0] + self.main_3d_img_W/2),
                                                    int(positions_main_img_1080p_pixel_coordinate[1] + self.main_3d_img_H/2)]
                        positions_main_img_1080p = [min(max(20,positions_main_img_1080p_[0]), self.main_3d_img_W-20),
                                                    min(max(20,positions_main_img_1080p_[1]), self.main_3d_img_H-20)]
                        positions_main_img = [int(positions_main_img_1080p[0] * self.main_panel_img_W / self.main_3d_img_W),
                                              int(positions_main_img_1080p[1] * self.main_panel_img_H / self.main_3d_img_H)]

                    else:
                        positions_main_img = [-1, -1]
                        positions_main_img_1080p = [-1, -1]

                else:
                    positions_main_img = [-1, -1]
                    positions_main_img_1080p = [-1, -1]

                if self.Version_mode == 2 or self.Version_mode == 3:
                    if is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topleft):
                        if self.main_panel_appear_flag is True:
                            self.main_panel_appear_time = time.time()
                            self.main_panel_appear_flag = False

                        if time.time() - self.main_panel_appear_time > self.delay_main_panel_appear:
                            self.gui_display = True
                            self.gui_side_left = True
                            self.gui_display_time = time.time()
                            self.gui_display_time_delay = 0

                    elif is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topright):
                        if self.main_panel_appear_flag is True:
                            self.main_panel_appear_time = time.time()
                            self.main_panel_appear_flag = False

                        if time.time() - self.main_panel_appear_time > self.delay_main_panel_appear:
                            self.gui_display = True
                            self.gui_side_left = False
                            self.gui_display_time = time.time()
                            self.gui_display_time_delay = 0
                    else:
                        self.main_panel_appear_flag = True

                elif self.Version_mode == 1:
                    if is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topleft) or \
                            is_pos_in_bbox(positions_main_img, self.edge_bbox_topleft):
                        if self.gui_display == False and self.main_panel_appear_flag_right is True:  # todo
                            if self.main_panel_appear_flag_left is True:
                                self.main_panel_appear_time_left = time.time()
                                self.main_panel_appear_flag_left = False

                            if self.main_panel_appear_flag_left == False and \
                                    time.time() - self.main_panel_appear_time_left > self.delay_main_panel_appear:
                                self.gui_display = True
                                self.gui_side_left = True
                                self.gui_display_time = time.time()
                                self.gui_display_time_delay = 0
                                # if is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topleft):
                                #     self.aux_control_flag = True
                                #     self.main_control_flag = False
                                # elif is_pos_in_bbox(positions_main_img, self.edge_bbox_topleft):
                                #     self.aux_control_flag = False
                                #     self.main_control_flag = True
                                if is_pos_in_bbox(positions_main_img, self.edge_bbox_topleft):
                                    self.aux_control_flag = False
                                    self.main_control_flag = True
                                else:
                                    exist, idx = is_multipos_in_bbox_with_idx(positions_aux_img, self.edge_bbox_topleft)
                                    self.aux_control_flag = True
                                    self.aux_control_idx = idx
                                    self.main_control_flag = False
                    else:
                        self.main_panel_appear_flag_left = True

                    if is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topright) or \
                            is_pos_in_bbox(positions_main_img, self.edge_bbox_topright):
                        if self.gui_display == False and self.main_panel_appear_flag_left is True:  # todo
                            if self.main_panel_appear_flag_right is True:
                                self.main_panel_appear_time_right = time.time()
                                self.main_panel_appear_flag_right = False

                            if self.main_panel_appear_flag_right == False and \
                                    time.time() - self.main_panel_appear_time_right > self.delay_main_panel_appear:
                                self.gui_display = True
                                self.gui_side_left = False
                                self.gui_display_time = time.time()
                                self.gui_display_time_delay = 0
                                # if is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topright):
                                #     self.aux_control_flag = True
                                #     self.main_control_flag = False
                                # elif is_pos_in_bbox(positions_main_img, self.edge_bbox_topright):
                                #     self.aux_control_flag = False
                                #     self.main_control_flag = True
                                if is_pos_in_bbox(positions_main_img, self.edge_bbox_topright):
                                    self.aux_control_flag = False
                                    self.main_control_flag = True
                                else:
                                    exist, idx = is_multipos_in_bbox_with_idx(positions_aux_img, self.edge_bbox_topright)
                                    self.aux_control_flag = True
                                    self.main_control_flag = False
                                    self.aux_control_idx = idx
                    else:
                        self.main_panel_appear_flag_right = True
                    # if is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topleft) or \
                    #         is_pos_in_bbox(positions_main_img, self.edge_bbox_topleft):
                    #     if self.main_panel_appear_flag is True:
                    #         self.main_panel_appear_time = time.time()
                    #         self.main_panel_appear_flag = False
                    #
                    #     if time.time() - self.main_panel_appear_time > self.delay_main_panel_appear:
                    #         self.gui_display = True
                    #         self.gui_side_left = True
                    #         self.gui_display_time = time.time()
                    #         self.gui_display_time_delay = 0
                    #
                    # elif is_multipos_in_bbox(positions_aux_img, self.edge_bbox_topright) or \
                    #         is_pos_in_bbox(positions_main_img, self.edge_bbox_topright):
                    #     if self.main_panel_appear_flag is True:
                    #         self.main_panel_appear_time = time.time()
                    #         self.main_panel_appear_flag = False
                    #
                    #     if time.time() - self.main_panel_appear_time > self.delay_main_panel_appear:
                    #         self.gui_display = True
                    #         self.gui_side_left = False
                    #         self.gui_display_time = time.time()
                    #         self.gui_display_time_delay = 0
                    # else:
                    #     self.main_panel_appear_flag = True

                time_gui_start = time.time()
                # self.gui_side_left = False

                if self.gui_display is True:
                    if self.gui_side_left is True:
                        if self.Version_mode == 2 or self.Version_mode == 3:
                            self.img_right_src_RGB_1080p = self.draw_img_GUI_left_osfa(self.img_right_src_RGB_1080p,
                                                           position=positions_aux_img, position_main=positions_main_img)

                        elif self.Version_mode == 1:
                            positions_main_img_one = positions_main_img
                            # positions_aux_img_one = positions_aux_img
                            positions_aux_img_one = []
                            if self.aux_control_flag is True:
                                positions_main_img_one = [-1, -1]
                                # positions_aux_img_one = []
                                for i in range(len(self.pred_position_aux)):
                                    position_aux_img_one = [-1, -1]
                                    positions_aux_img_one.append(position_aux_img_one)
                                positions_aux_img_one[self.aux_control_idx] = positions_aux_img[self.aux_control_idx]

                            elif self.main_control_flag is True:
                                # positions_aux_img_one = []
                                for i in range(len(self.pred_position_aux)):
                                    position_aux_img_one = [-1, -1]
                                    positions_aux_img_one.append(position_aux_img_one)

                            else:
                                positions_aux_img_one = positions_aux_img.copy()

                            self.img_right_src_RGB_1080p = self.draw_img_GUI_left_pure(self.img_right_src_RGB_1080p,
                                                           position=positions_aux_img_one, position_main=positions_main_img_one)

                            # positions_main_img_one = positions_main_img
                            # positions_aux_img_one = positions_aux_img
                            # if self.aux_control_flag is True:
                            #     positions_main_img_one = [-1, -1]
                            # elif self.main_control_flag is True:
                            #     positions_aux_img_one = []
                            #     for i in range(len(self.pred_position_aux)):
                            #         position_aux_img_one = [-1, -1]
                            #         positions_aux_img_one.append(position_aux_img_one)
                            # self.img_right_src_RGB_1080p = self.draw_img_GUI_left_pure(self.img_right_src_RGB_1080p,
                            #                                position=positions_aux_img_one, position_main=positions_main_img_one)

                            # self.img_right_src_RGB_1080p = self.draw_img_GUI_left_pure(self.img_right_src_RGB_1080p,
                            #                                position=positions_aux_img, position_main=positions_main_img)

                    else:
                        if self.Version_mode == 2 or self.Version_mode == 3:
                            self.img_right_src_RGB_1080p = self.draw_img_GUI_right_osfa(self.img_right_src_RGB_1080p,
                                                        position=positions_aux_img, position_main=positions_main_img)
                        elif self.Version_mode == 1:
                            positions_main_img_one = positions_main_img
                            # positions_aux_img_one = positions_aux_img
                            positions_aux_img_one = []
                            if self.aux_control_flag is True:
                                positions_main_img_one = [-1, -1]
                                for i in range(len(self.pred_position_aux)):
                                    position_aux_img_one = [-1, -1]
                                    positions_aux_img_one.append(position_aux_img_one)
                                positions_aux_img_one[self.aux_control_idx] = positions_aux_img[self.aux_control_idx]

                            elif self.main_control_flag is True:
                                # positions_aux_img_one = []
                                for i in range(len(self.pred_position_aux)):
                                    position_aux_img_one = [-1, -1]
                                    positions_aux_img_one.append(position_aux_img_one)
                            else:
                                positions_aux_img_one = positions_aux_img.copy()
                            self.img_right_src_RGB_1080p = self.draw_img_GUI_right_pure(self.img_right_src_RGB_1080p,
                                                        position=positions_aux_img_one, position_main=positions_main_img_one)

                            # positions_main_img_one = positions_main_img
                            # positions_aux_img_one = positions_aux_img
                            # if self.aux_control_flag is True:
                            #     positions_main_img_one = [-1, -1]
                            # elif self.main_control_flag is True:
                            #     positions_aux_img_one = []
                            #     for i in range(len(self.pred_position_aux)):
                            #         position_aux_img_one = [-1, -1]
                            #         positions_aux_img_one.append(position_aux_img_one)
                            # self.img_right_src_RGB_1080p = self.draw_img_GUI_right_pure(self.img_right_src_RGB_1080p,
                            #                             position=positions_aux_img_one, position_main=positions_main_img_one)

                            # self.img_right_src_RGB_1080p = self.draw_img_GUI_right_pure(self.img_right_src_RGB_1080p,
                            #                             position=positions_aux_img, position_main=positions_main_img)

                    if self.Version_mode == 0:
                        self.img_right_src_RGB_1080p = self.draw_img_GUI_left_none(self.img_right_src_RGB_1080p,
                                                                                   position=positions_aux_img)
                else:
                    self.uterus_panel_area_exit = True
                    self.gui_main_panel_time[0, 0] = time.time()
                    self.aux_control_flag = False
                    self.main_control_flag = False
                    self.aux_control_idx = -1


                time_gui = time.time()
                print('------ gui_time', time_gui - time_gui_start)

                # display the tool with a color point: auxliary tools + main tools
                # img_right_src_RGB_ = self.img_right_src_RGB_1080p.copy()
                # self.img_right_src_RGB = cv2.resize(img_right_src_RGB_, (self.main_panel_img_W, self.main_panel_img_H))
                self.img_right_src_RGB = cv2.resize(self.img_right_src_RGB_1080p, (self.main_panel_img_W, self.main_panel_img_H))

                time_draw_points1 = time.time()

                for i in range(len(positions_aux_img)):
                    if positions_aux_img[i][0] >= 0 and positions_aux_img[i][1] >= 0:
                        if self.Version_mode != 0:
                            self.img_right_src_RGB_1080p = main_aux_point_icon_plot_1080p(self.img_right_src_RGB_1080p, (positions_aux_img_1080p[i][0], positions_aux_img_1080p[i][1]), self.aux_point_icon, (self.main_3d_img_W, self.main_3d_img_H))
                        # self.img_right_src_RGB = main_aux_point_icon_plot(self.img_right_src_RGB, (positions_aux_img[i][0], positions_aux_img[i][1]), self.aux_point_icon, (self.main_panel_img_W, self.main_panel_img_H))
                        self.img_right_src_RGB = cv2.circle(self.img_right_src_RGB, (positions_aux_img[i][0], positions_aux_img[i][1]), 4, (123, 238, 253), 16)

                if positions_main_img[0] >= 0 and positions_main_img[1] >= 0:
                    if self.Version_mode != 0:
                        self.img_right_src_RGB_1080p = main_aux_point_icon_plot_1080p(self.img_right_src_RGB_1080p, (positions_main_img_1080p[0], positions_main_img_1080p[1]), self.main_point_icon, (self.main_3d_img_W, self.main_3d_img_H))
                    # self.img_right_src_RGB = main_aux_point_icon_plot(self.img_right_src_RGB, (positions_main_img[0], positions_main_img[1]), self.main_point_icon, (self.main_panel_img_W, self.main_panel_img_H))
                    self.img_right_src_RGB = cv2.circle(self.img_right_src_RGB, (positions_main_img[0], positions_main_img[1]), 4, (253, 238, 123), 16)

                time_draw_points2 = time.time()
                print('*********** time_draw_points1 ***********', time_draw_points2 - time_draw_points1)

                # display the reference white/yellow bbox
                if (self.whether_visualize and not Manual_Mode and not self.mode == 'Uterus' and not self.mode == 'Dominant') or self.whether_visualize_in_setting:
                    self.img_right_src_RGB = self.draw_img_Refer_BBox(self.img_right_src_RGB, bbox=self.white_bbox_refer_src, color=(255, 255, 255))
                    self.img_right_src_RGB = self.draw_img_Refer_BBox(self.img_right_src_RGB, bbox=self.yellow_bbox_refer_src, color=(255, 255, 0))

                    self.img_right_src_RGB_1080p = self.draw_img_Refer_BBox(self.img_right_src_RGB_1080p, bbox=self.white_bbox_refer_src, color=(255, 255, 255))
                    self.img_right_src_RGB_1080p = self.draw_img_Refer_BBox(self.img_right_src_RGB_1080p, bbox=self.yellow_bbox_refer_src, color=(255, 255, 0))

                # self.spanslider_z.setRange(int(self.mid_depth - self.mid_range/2), int(self.mid_depth + self.mid_range/2))
                # display the depth into the image
                if self.Zoomin_Mode or self.Zoomout_Mode:
                    DepthRange_display = f'Depth Range: {self.mid_depth - self.mid_range/2:.1f} ~ {self.mid_depth + self.mid_range/2:.1f} mm'
                    cv2.putText(self.img_right_src_RGB_1080p, DepthRange_display, (450, 100), 0, 2, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                time_draw_refer = time.time()
                # publish the main tool index
                self.main_tool_id_pub.publish(Int32(self.main_tool_id))
                self.system_version_pub.publish(Int32(self.Version_mode))
                # publish the bbox scale factor
                self.scale_bbox_pub.publish(Float64(self.SettingBoxValue))
                # publish the mid depth information
                self.mid_depth_pub.publish(Float64(self.mid_depth))
                self.mid_range_pub.publish(Float64(self.mid_range))
                # publish the reference baseframeX direction
                baseframex_msg = Vector3(self.robot.bRc_init[0, 0], self.robot.bRc_init[1, 0], self.robot.bRc_init[2, 0])
                self.base_framex_pub.publish(baseframex_msg)

                time_publish = time.time()

                self.show_img()  # todo: test the time cost
                time_show1 = time.time()
                # self.Show_L_Img_1080p()
                time_showL = time.time()
                self.Show_R_Img_1080p()
                time_finish = time.time()
                self.loop_rate.sleep()

                print('show time cost: {}/{}/{}'.format(time_show1 - time_publish, time_showL - time_show1, time_finish - time_showL))

                # UI update
                # For GUI Interface Monitor
                if self.gui_display != self.gui_display_pre:
                    self.statusled_GUI.click()
                self.lineEdit_10.setText(self.mode) # GUI mode
                # For Laparoscope Holder
                self.lineEdit.setText(f'x {self.SettingCspeedValue:.2f}')
                if self.Emergency_Mode != self.Emergency_Mode_pre:
                    self.statusled_LaparoscopeHolder.click()

                Laparoscope_Status = 'None'
                if self.Return_Track_Mode: #  and self.mode == 'Camera'
                    if self.target_status is False:
                        Laparoscope_Status = 'Global'
                    else:
                        Laparoscope_Status = 'Global-Keep'

                if self.Zoomin_Mode:
                    Laparoscope_Status = 'Zoom-in'
                if self.Zoomout_Mode:
                    Laparoscope_Status = 'Zoom-out'
                if self.Track_z_Mode:
                    Laparoscope_Status = 'Track'
                self.lineEdit_4.setText(Laparoscope_Status)

                if Manual_Mode:
                    Laparoscope_Mode = 'Manual'
                else:
                    Laparoscope_Mode = 'Automatic'
                if self.mode == 'Camera':
                    Laparoscope_Mode = 'GUI'
                self.lineEdit_3.setText(Laparoscope_Mode)

                if self.SettingLap0:
                    self.lineEdit_5.setText('0 deg')
                else:
                    self.lineEdit_5.setText('30 deg')

                self.lineEdit_6.setText(str(self.main_tool_id))
                self.show_main_tool()
                self.dial.setSliderPosition(50+self.SettingRotationValue)

                # For Uterus Manipulator
                self.lineEdit_8.setText(f'x {self.ControlUspeedValue:.2f}') # uterus
                Uterus_manipulator_Mode = 'Manual'
                if self.UterusAuto_Mode:
                    Uterus_manipulator_Mode = 'Automatic'
                if self.mode == 'Uterus':
                    Uterus_manipulator_Mode = 'GUI'
                self.lineEdit_7.setText(Uterus_manipulator_Mode) # Mode
                # TODO:
                Uterus_manipulator_Status = 'None'
                if self.UterusUp_Mode and self.mode == 'Uterus':
                    Uterus_manipulator_Status = 'Up'
                if self.UterusDown_Mode and self.mode == 'Uterus':
                    Uterus_manipulator_Status = 'Down'
                if self.UterusLeft_Mode and self.mode == 'Uterus':
                    Uterus_manipulator_Status = 'Left'
                if self.UterusRight_Mode and self.mode == 'Uterus':
                    Uterus_manipulator_Status = 'Right'
                if self.UterusInsert_Mode and self.mode == 'Uterus':
                    Uterus_manipulator_Status = 'Insert'
                if self.UterusRetract_Mode and self.mode == 'Uterus':
                    Uterus_manipulator_Status = 'Retract'
                self.lineEdit_9.setText(Uterus_manipulator_Status) # Mode
                # self.lineEdit_14 # uterus time

                self.lineEdit_uterus_pitch.setText(f'{self.uterus_angle_pitch}')
                if self.uterus_angle_pitch > 64 or self.uterus_angle_pitch < -19:
                    self.lineEdit_uterus_pitch.setStyleSheet("color:red")
                # elif self.uterus_angle_pitch > 62 or self.uterus_angle_pitch < -18:
                #     self.lineEdit_uterus_pitch.setStyleSheet("color:blue")
                else:
                    self.lineEdit_uterus_pitch.setStyleSheet("color:black")

                self.lineEdit_uterus_yaw.setText(f'{self.uterus_angle_yaw}')
                if self.uterus_angle_yaw > 25 or self.uterus_angle_yaw < -25:
                    self.lineEdit_uterus_yaw.setStyleSheet("color:red")
                # elif self.uterus_angle_yaw > 23 or self.uterus_angle_yaw < -23:
                #     self.lineEdit_uterus_yaw.setStyleSheet("color:blue")
                else:
                    self.lineEdit_uterus_yaw.setStyleSheet("color:black")

                self.lineEdit_uterus_insertion.setText(f'{self.uterus_angle_insertion}')
                if self.uterus_angle_insertion > 48:   # or self.uterus_angle_insertion < 2:
                    self.lineEdit_uterus_insertion.setStyleSheet("color:red")
                # elif self.uterus_angle_insertion > 23 or self.uterus_angle_insertion < 5:
                #     self.lineEdit_uterus_insertion.setStyleSheet("color:blue")
                else:
                    self.lineEdit_uterus_insertion.setStyleSheet("color:black")

                self.lineEdit_uterus_rotation.setText(f'{self.uterus_angle_rotation}')
                if self.uterus_angle_rotation > 24:  # or self.uterus_angle_rotation < 2:
                    self.lineEdit_uterus_rotation.setStyleSheet("color:red")
                else:
                    self.lineEdit_uterus_rotation.setStyleSheet("color:black")

                self.lineEdit_uterus_grasp.setText(f'{self.uterus_angle_grasp}')
                if self.uterus_angle_grasp > 44:  # or self.uterus_angle_grasp < 2:
                    self.lineEdit_uterus_grasp.setStyleSheet("color:red")
                else:
                    self.lineEdit_uterus_grasp.setStyleSheet("color:black")

                System_version_Status = f'V-{self.Version_mode}'
                self.lineEdit_Version.setText(System_version_Status)

                System_initDepth_Status = f' Ready ' if System_initDepth_flag else f' None '
                self.lineEdit_InitialDepth.setText(System_initDepth_Status)
                self.lineEdit_WorkspaceRange.setText(str(int(self.Workspace_Inter)))
                self.lineEdit_RCM2CAM.setText(str(int(self.dist_r2c)))

                self.lineEdit_SurgicalPhase.setText('Phase' + str(int(self.Phase_Cur)))

                # Main tool information
                self.lineEdit_12.setText(str(positions_main_img[0])) # PosX info
                self.lineEdit_11.setText(str(positions_main_img[1])) # PosY info
                self.lineEdit_13.setText(str(int(self.tool_depth*1000))) # depth info
                self.lineEdit_14.setText(str(int(self.mid_depth))) # mid depth info

                # Joystick information
                self.joystick_Laparoscope.setPosXY(-actions_img[0]*80, actions_img[1]*80)
                self.joystick_Uterus_manipulator.setPosXY(-uterus_actions_img[0]*80, uterus_actions_img[1]*80)

                self.slider_Laparoscope.setSliderPosition((actions_img[2] + 1)*50) # 50: 0
                self.slider_Uterus_manipulator.setSliderPosition((uterus_actions_img[2] + 1)*50)

                # Depth information and the x y position display
                # self.spanslider_x.setRange(int((self.white_bbox_refer_src.cx - self.white_bbox_refer_src.w/2)*100), int((self.white_bbox_refer_src.cx + self.white_bbox_refer_src.w/2)*100))
                self.spanslider_x.setLowerPosition(int((self.white_bbox_refer_src.cx - self.white_bbox_refer_src.w/2)*100))
                self.spanslider_x.setUpperPosition(int((self.white_bbox_refer_src.cx + self.white_bbox_refer_src.w/2)*100))
                self.spanslider_y.setLowerPosition(int((self.white_bbox_refer_src.cy - self.white_bbox_refer_src.h/2)*100))
                self.spanslider_y.setUpperPosition(int((self.white_bbox_refer_src.cy + self.white_bbox_refer_src.h/2)*100))
                self.spanslider_z.setLowerPosition(int(self.mid_depth - self.mid_range/2))
                self.spanslider_z.setUpperPosition(int(self.mid_depth + self.mid_range/2))
                # self.spanslider_z.setRange(int(self.mid_depth - self.mid_range/2), int(self.mid_depth + self.mid_range/2))

                # display the depth into the image


                # record the reference, for detect the status change
                self.gui_display_pre = self.gui_display
                self.Emergency_Mode_pre = self.Emergency_Mode

                # print('dist1: {}, dist2: {}'.format(dist1, dist2))
                # print('normal_plane1: {}, normal_plane1: {}'.format(normal_plane1, normal_plane2))
                # print('proj1: {}, proj2: {}'.format(proj1, proj2))
                # print('LR_angle: {}, UD_angle: {}'.format(self.LR_angle, self.UD_angle))

                QApplication.processEvents()
                time_end = time.time()
                # record the necessary info: self.blackbox_pub
                # TODO
                # self.lineEdit_14.setText(f'{self.gui_main_panel_time_delay[0, 0]:.2f}')
                self.blackbox_info_msg.header.stamp = rospy.Time.now()
                self.blackbox_info_msg.system_version = self.Version_mode
                self.blackbox_info_msg.init_workspace = System_initDepth_flag
                self.blackbox_info_msg.workspace_range = self.Workspace_Inter
                self.blackbox_info_msg.rcm2cam = self.dist_r2c
                self.blackbox_info_msg.gui_activate = self.gui_display
                self.blackbox_info_msg.gui_status = self.mode
                self.blackbox_info_msg.lap_emergency_stop = self.Emergency_Mode
                self.blackbox_info_msg.lap_ctl_speed = self.SettingCspeedValue
                self.blackbox_info_msg.lap_mode = Laparoscope_Mode
                self.blackbox_info_msg.lap_gui_status = Laparoscope_Status
                self.blackbox_info_msg.main_tool_id = self.main_tool_id
                self.blackbox_info_msg.lap_type = 0 if self.SettingLap0 else 30
                self.blackbox_info_msg.lap_action = np.array([self.pred_action.x, self.pred_action.y, self.pred_action.z])
                self.blackbox_info_msg.lap_misori = self.SettingRotationValue
                self.blackbox_info_msg.track_box_scale = self.SettingBoxValue
                self.blackbox_info_msg.mid_depth = self.mid_depth
                self.blackbox_info_msg.mid_range = self.mid_range
                self.blackbox_info_msg.uterus_mp_emergency_stop = False
                self.blackbox_info_msg.uterus_mp_ctl_speed = self.SettingUspeedValue
                self.blackbox_info_msg.uterus_mp_mode = Uterus_manipulator_Mode
                self.blackbox_info_msg.uterus_mp_gui_status = Uterus_manipulator_Status
                self.blackbox_info_msg.uterus_action = np.array(self.action_uterus)
                self.blackbox_info_msg.uterus_tip_usage = self.uterus_tip_usage
                self.blackbox_info_msg.uterus_angle = np.array(self.uterus_angle_all, dtype=np.int32)
                self.blackbox_info_msg.uterus_angle_control = np.array(self.uterus_angle_control)
                self.blackbox_info_msg.uterus_target_angle = np.array(self.uterus_target_angle)
                self.blackbox_info_msg.pitch_error = self.pitch_error
                self.blackbox_info_msg.yaw_error = self.yaw_error
                self.blackbox_info_msg.cVc = self.vision_params.Parameter
                self.blackbox_info_msg.err = self.vision_params.ImageError[0:2]
                bTc_cur = self.robot.get_bTc(self.joint_pos)
                self.blackbox_info_msg.bTc = np.reshape(bTc_cur, (-1,))
                self.blackbox_pub.publish(self.blackbox_info_msg)
                time_end_blackbox = time.time()

                print('one loop time cost: {}/{}/{}/{}/{}/{}/{}'.format(time_end - time_finish,
                                                                        time_finish - time_publish,
                                                                        time_publish - time_draw_refer,
                                                                        time_draw_refer - time_gui,
                                                                        time_gui - time_draw_1, time_draw_1 - time_cvt,
                                                                        time_cvt - time_start))
                print('total time: ', time_end - time_start)
                # print('blackbox pub time: ', time_end_blackbox - time_end)
        else:
            # self.textEdit.setText('No camera connected!')
            msg = QMessageBox.warning(self, 'warning', 'No camera connected!', buttons=QMessageBox.Ok)
            print('No camera connected!')

    def pushButton_Setup(self):
        """
        Use the Joystick to control the settings
        """
        self.mode = 'Setting'

        if self.Version_mode == 1 or self.Version_mode == 0:
            self.setting_button_index = self.setting_button_index - self.joy_src.axes[1] * (
                        time.time() - self.cur_time_button) * 3
            self.cur_time_button = time.time()
            if self.setting_button_index >= 6.9:
                self.setting_button_index = 6.9
            if self.setting_button_index <= 0.1:
                self.setting_button_index = 0.1
            # print(self.setting_button_index)

            self.SettingLap = True if int(self.setting_button_index) == 1 else False
            self.SettingHand = True if int(self.setting_button_index) == 2 else False
            self.SettingUspeed = True if int(self.setting_button_index) == 3 else False
            self.SettingCspeed = True if int(self.setting_button_index) == 4 else False
            self.SettingRotation = True if int(self.setting_button_index) == 5 else False
            self.SettingTipUsage = True if int(self.setting_button_index) == 6 else False

            # ---------- setting action --------------
            if self.SettingLap:
                if self.joy_src.buttons[2] == 1 and self.SettingLap0_flag is True:
                    self.SettingLap0 = not self.SettingLap0
                    self.SettingLap0_flag = False
                if self.joy_src.buttons[2] == 0 and self.SettingLap0_flag is False:
                    self.SettingLap0_flag = True
                # print('Setting SettingLap0_flag: ', self.SettingLap0_flag)
                self.SettingLap = False

            elif self.SettingHand:
                if self.joy_src.buttons[2] == 1 and self.SettingHand_flag is True:
                    self.SettingHandLeft = not self.SettingHandLeft
                    self.SettingHand_flag = False
                if self.joy_src.buttons[2] == 0 and self.SettingHand_flag is False:
                    self.SettingHand_flag = True
                # print('Setting SettingHand_flag: ', self.SettingHand_flag, self.SettingHandLeft)
                self.SettingHand = False

            elif self.SettingUspeed:
                # Uterus speed
                self.SettingUspeedValue = self.SettingUspeedValue - self.joy_src.axes[2] * (
                            time.time() - self.cur_time)  # * 0.1
                self.cur_time = time.time()
                if self.SettingUspeedValue >= 2.0:
                    self.SettingUspeedValue = 2.0
                if self.SettingUspeedValue <= 0.5:
                    self.SettingUspeedValue = 0.5
                self.SettingUspeed = False

            elif self.SettingCspeed:
                # camera speed
                self.SettingCspeedValue = self.SettingCspeedValue - self.joy_src.axes[2] * (
                            time.time() - self.cur_time)  # * 0.1
                self.cur_time = time.time()
                if self.SettingCspeedValue >= 4.0:
                    self.SettingCspeedValue = 4.0
                if self.SettingCspeedValue <= 0.1:
                    self.SettingCspeedValue = 0.1
                self.SettingCspeed = False

            elif self.SettingRotation:
                # camera degree/rotation: ori, 0 --> -45 ~ 45 degree
                self.SettingRotationValue = self.SettingRotationValue - self.joy_src.axes[2] * (
                            time.time() - self.cur_time) * 2  # * 0.1  5
                self.cur_time = time.time()
                if self.SettingRotationValue >= 45:
                    self.SettingRotationValue = 45
                if self.SettingRotationValue <= -45:
                    self.SettingRotationValue = -45
                self.SettingRotation = False

            if self.SettingTipUsage:
                if self.joy_src.buttons[2] == 1 and self.SettingTipUsage_flag is True:
                    if self.SettingTipUsage0 == 'True':
                        self.SettingTipUsage0 = 'False'
                    else:
                        self.SettingTipUsage0 = 'True'
                    self.SettingTipUsage_flag = False
                if self.joy_src.buttons[2] == 0 and self.SettingTipUsage_flag is False:
                    self.SettingTipUsage_flag = True
                self.SettingTipUsage = False

        # if self.Version_mode == 1 or self.Version_mode == 0:
        #     self.setting_button_index = self.setting_button_index - self.joy_src.axes[1] * (
        #                 time.time() - self.cur_time_button) * 3
        #     self.cur_time_button = time.time()
        #     if self.setting_button_index >= 7.9:
        #         self.setting_button_index = 7.9
        #     if self.setting_button_index <= 0.1:
        #         self.setting_button_index = 0.1
        #     # print(self.setting_button_index)
        # 
        #     self.SettingLap = True if int(self.setting_button_index) == 1 else False
        #     self.SettingHand = True if int(self.setting_button_index) == 2 else False
        #     self.SettingUspeed = True if int(self.setting_button_index) == 3 else False
        #     self.SettingTipUsage = True if int(self.setting_button_index) == 4 else False
        # 
        #     # ---------- setting action --------------
        #     if self.SettingLap:
        #         if self.joy_src.buttons[2] == 1 and self.SettingLap0_flag is True:
        #             self.SettingLap0 = not self.SettingLap0
        #             self.SettingLap0_flag = False
        #         if self.joy_src.buttons[2] == 0 and self.SettingLap0_flag is False:
        #             self.SettingLap0_flag = True
        #         # print('Setting SettingLap0_flag: ', self.SettingLap0_flag)
        #         self.SettingLap = False
        # 
        #     elif self.SettingHand:
        #         if self.joy_src.buttons[2] == 1 and self.SettingHand_flag is True:
        #             self.SettingHandLeft = not self.SettingHandLeft
        #             self.SettingHand_flag = False
        #         if self.joy_src.buttons[2] == 0 and self.SettingHand_flag is False:
        #             self.SettingHand_flag = True
        #         # print('Setting SettingHand_flag: ', self.SettingHand_flag, self.SettingHandLeft)
        #         self.SettingHand = False
        # 
        #     elif self.SettingUspeed:
        #         # Uterus speed
        #         self.SettingUspeedValue = self.SettingUspeedValue - self.joy_src.axes[2] * (
        #                     time.time() - self.cur_time)  # * 0.1
        #         self.cur_time = time.time()
        #         if self.SettingUspeedValue >= 2.0:
        #             self.SettingUspeedValue = 2.0
        #         if self.SettingUspeedValue <= 0.5:
        #             self.SettingUspeedValue = 0.5
        #         self.SettingUspeed = False
        # 
        #     if self.SettingTipUsage:
        #         if self.joy_src.buttons[2] == 1 and self.SettingTipUsage_flag is True:
        #             if self.SettingTipUsage0 == 'True':
        #                 self.SettingTipUsage0 = 'False'
        #             else:
        #                 self.SettingTipUsage0 = 'True'
        #             self.SettingTipUsage_flag = False
        #         if self.joy_src.buttons[2] == 0 and self.SettingTipUsage_flag is False:
        #             self.SettingTipUsage_flag = True
        #         self.SettingTipUsage = False

        elif self.Version_mode == 2 or self.Version_mode == 3:
            self.setting_button_index = self.setting_button_index - self.joy_src.axes[1] * (
                        time.time() - self.cur_time_button) * 3
            self.cur_time_button = time.time()
            if self.setting_button_index >= 8.9:
                self.setting_button_index = 8.9
            if self.setting_button_index <= 0.1:
                self.setting_button_index = 0.1
            # print(self.setting_button_index)

            self.SettingLap = True if int(self.setting_button_index) == 1 else False
            self.SettingHand = True if int(self.setting_button_index) == 2 else False
            self.SettingUspeed = True if int(self.setting_button_index) == 3 else False
            self.SettingCspeed = True if int(self.setting_button_index) == 4 else False
            self.SettingBox = True if int(self.setting_button_index) == 5 else False
            self.SettingDepth = True if int(self.setting_button_index) == 6 else False
            self.SettingRotation = True if int(self.setting_button_index) == 7 else False
            self.SettingTipUsage = True if int(self.setting_button_index) == 8 else False

            self.whether_visualize_in_setting = True if int(self.setting_button_index) == 5 else False

            # self.setting_panel_count[0, int(self.setting_button_index)] = 1e12

            # ---------- setting action --------------
            if self.SettingLap:
                if self.joy_src.buttons[2] == 1 and self.SettingLap0_flag is True:
                    self.SettingLap0 = not self.SettingLap0
                    self.SettingLap0_flag = False
                if self.joy_src.buttons[2] == 0 and self.SettingLap0_flag is False:
                    self.SettingLap0_flag = True
                # print('Setting SettingLap0_flag: ', self.SettingLap0_flag)
                self.SettingLap = False

            elif self.SettingHand:
                # if self.joy_src.buttons[2] == 1 and self.SettingHand_flag is True:
                if self.joy_src.buttons[2] == 1:
                    self.SettingGlobal = True
                    self.SettingGlobal0 = 'True'
                    # self.SettingHandLeft = not self.SettingHandLeft
                    self.SettingHand_flag = False
                # if self.joy_src.buttons[2] == 0 and self.SettingHand_flag is False:
                if self.joy_src.buttons[2] == 0:
                    # self.SettingHand_flag = True
                    self.SettingGlobal0 = 'False'
                # print('Setting SettingHand_flag: ', self.SettingHand_flag, self.SettingHandLeft)
                self.SettingHand = False

            elif self.SettingUspeed:
                # Uterus speed
                self.SettingUspeedValue = self.SettingUspeedValue - self.joy_src.axes[2] * (
                            time.time() - self.cur_time)  # * 0.1
                self.cur_time = time.time()
                if self.SettingUspeedValue >= 2.0:
                    self.SettingUspeedValue = 2.0
                if self.SettingUspeedValue <= 0.5:
                    self.SettingUspeedValue = 0.5
                self.SettingUspeed = False

            elif self.SettingCspeed:
                # camera speed
                self.SettingCspeedValue = self.SettingCspeedValue - self.joy_src.axes[2] * (time.time() - self.cur_time)  # * 0.1
                self.cur_time = time.time()
                if self.SettingCspeedValue >= 3.0:
                    self.SettingCspeedValue = 3.0
                if self.SettingCspeedValue <= 0.1:
                    self.SettingCspeedValue = 0.1
                self.SettingCspeed = False

            elif self.SettingBox:
                self.SettingBoxValue = self.SettingBoxValue + self.joy_src.axes[2] * (time.time() - self.cur_time) * 0.1
                self.cur_time = time.time()
                if self.SettingBoxValue >= 1.9:
                    self.SettingBoxValue = 1.9
                if self.SettingBoxValue <= 0.1:
                    self.SettingBoxValue = 0.1
                # by default: the scale is the 1.0, need to in range(0.0, 2.0)
                self.SettingBox = False

            elif self.SettingDepth:     # todo setting the rate
                # center position: ori, 75 --> 50 ~ 100mm
                self.SettingDepthRate = self.SettingDepthRate - self.joy_src.axes[2] * (time.time() - self.cur_time)
                self.cur_time = time.time()
                if self.SettingDepthRate >= 3.0:
                    self.SettingDepthRate = 3.0
                if self.SettingDepthRate <= 0.5:
                    self.SettingDepthRate = 0.5
                self.SettingDepth = False

            elif self.SettingRotation:
                # camera degree/rotation: ori, 0 --> -45 ~ 45 degree
                self.SettingRotationValue = self.SettingRotationValue - self.joy_src.axes[2] * (
                            time.time() - self.cur_time) * 2  # * 0.1  5
                self.cur_time = time.time()
                if self.SettingRotationValue >= 45:
                    self.SettingRotationValue = 45
                if self.SettingRotationValue <= -45:
                    self.SettingRotationValue = -45
                self.SettingRotation = False

            if self.SettingTipUsage:
                if self.joy_src.buttons[2] == 1 and self.SettingTipUsage_flag is True:
                    if self.SettingTipUsage0 == 'True':
                        self.SettingTipUsage0 = 'False'
                    else:
                        self.SettingTipUsage0 = 'True'
                    self.SettingTipUsage_flag = False
                if self.joy_src.buttons[2] == 0 and self.SettingTipUsage_flag is False:
                    self.SettingTipUsage_flag = True
                self.SettingTipUsage = False

    def robot_correct_misorientation(self):
        """
        Move robot to correct the misorientation
        """
        self.actions = [0.0, 0.0, 0.0]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=0.1, flag=0)
        # self.vision_params_pub.publish(self.vision_params)
        cVc = self.vision_params.Parameter
        homo_delta = self.vision_params.ImageError[0:2]

        # drive the ur5 robot
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        if self.SettingGlobal and self.robot is not None:
            self.robot._init_global(self.joint_pos)
            self.SettingGlobal = False

        # modify the misorientation
        # TODO: IMPORTANT!!!
        delta_Rot = self.SettingRotationValue - self.SettingRotationPreValue
        self.robot.bRc_init = np.dot(RotZ(delta_Rot * np.pi / 180), self.robot.bRc_init)
        self.SettingRotationPreValue = self.SettingRotationValue
        # print("step1: cVc: ", cVc)
        # print("homo_delta: ", homo_delta)
        # print("----------------------------------------------bRc_init: \n", self.robot.bRc_init)
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        # print('delta_q: ', delta_q)
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    # @debug_class('MainWindow')
    # @pyqtSlot()
    def pushButton_Test_clicked(self):
        """
        Test for control the laparoscope by joystick
        :return:
        """
        time_ur_publish_0 = time.time()
        # calculate vision parameters(cVc/Errors) and publish the control signals
        # self.actions = [-self.joy_src.axes[0], -self.joy_src.axes[1], self.joy_src.axes[4]]
        self.actions = [-self.joy_src.axes[0] * self.image_xy_speed_coef,
                        -self.joy_src.axes[1] * self.image_xy_speed_coef,
                        -self.joy_src.axes[2] * self.depth_zz_speed_coef]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=0.1, flag=0)
        time_ur_publish_1 = time.time()
        # self.vision_params_pub.publish(self.vision_params)
        cVc = self.vision_params.Parameter
        homo_delta = self.vision_params.ImageError[0:2]

        time_ur_publish_2 = time.time()

        # drive the ur5 robot
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        if self.SettingGlobal and self.robot is not None:
            self.robot._init_global(self.joint_pos)
            self.SettingGlobal = False

        time_ur_publish_3 = time.time()
        # print("step1: cVc: ", cVc)
        # print("homo_delta: ", homo_delta)
        print("----------------------------------------------bTc_init: \n", self.robot.init_bTc)
        time_ur_publish_4 = time.time()
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        # print('delta_q: ', delta_q)
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

        time_ur_publish_done = time.time()
        print('^^^^^^^^^^^^^ur time cost: {}/{}/{}/{}/{}/{}'.format(time_ur_publish_done - time_ur_publish_4,
                                                                    time_ur_publish_4 - time_ur_publish_3,
                                                                    time_ur_publish_3 - time_ur_publish_2,
                                                                    time_ur_publish_2 - time_ur_publish_1,
                                                                    time_ur_publish_1 - time_ur_publish_0,
                                                                    time_ur_publish_done - time_ur_publish_0))

    def automatic_ctrl(self):
        """
        Control signals from predicted action
        :return:
        """
        # pass
        # calculate vision parameters(cVc/Errors) and publish the control signals
        if self.user_depth_flag is True or (self.user_depth_allow_zoomout_flag is False and self.pred_action.z > 0) \
                    or (self.user_depth_allow_zoomin_flag is False and self.pred_action.z < 0):
            self.actions = [-self.pred_action.x * self.image_xy_speed_coef,
                            -self.pred_action.y * self.image_xy_speed_coef,
                            0]
        else:  # no automatic depth control
            self.actions = [-self.pred_action.x * self.image_xy_speed_coef,
                            -self.pred_action.y * self.image_xy_speed_coef,
                            self.pred_action.z * self.depth_zz_speed_coef]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=self.tool_depth, flag=1)
        # self.vision_params_pub.publish(self.vision_params)
        cVc = self.vision_params.Parameter
        homo_delta = self.vision_params.ImageError[0:2]

        # drive the ur5 robot
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        if self.SettingGlobal and self.robot is not None:
            self.robot._init_global(self.joint_pos)
            self.SettingGlobal = False

        # modify the misorientation: make sure the speed command was send once per iteration to avoid damping effect
        # # TODO: IMPORTANT!!!
        # delta_Rot = self.SettingRotationValue - self.SettingRotationPreValue
        # self.robot.bRc_init = np.dot(RotZ(delta_Rot * np.pi / 180), self.robot.bRc_init)
        # self.SettingRotationPreValue = self.SettingRotationValue

        # print("step1: cVc: ", cVc)
        # print("homo_delta: ", homo_delta)
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        # print('delta_q: ', delta_q)
        if np.max(delta_q) >= 0.23 or np.min(delta_q) <= -0.23:
            print('********************************** over joint velocity limit: ', delta_q)
            # delta_q = np.zeros((6, 1))
            delta_q = delta_q / np.max(np.abs(delta_q)) * 0.23
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'

        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def semi_track_xy(self):
        """
        Control the FOV only use the 2D position information
        :return:
        """
        # pass
        # calculate vision parameters(cVc/Errors) and publish the control signals
        self.actions = [-self.pred_action.x * self.image_xy_speed_coef, -self.pred_action.y * self.image_xy_speed_coef,
                        0.0]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=self.tool_depth, flag=1)
        # self.vision_params_pub.publish(self.vision_params)
        cVc = self.vision_params.Parameter
        homo_delta = self.vision_params.ImageError[0:2]

        # drive the ur5 robot
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        if self.SettingGlobal and self.robot is not None:
            self.robot._init_global(self.joint_pos)
            self.SettingGlobal = False
        # print("step1: cVc: ", cVc)
        # print("homo_delta: ", homo_delta)
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        # print('delta_q: ', delta_q)
        # if np.max(delta_q) <= 1e-6:
        #     self.target_status = True # False means the target has already achieved
        #     self.Track_xy_Mode = False
        if np.max(delta_q) >= 0.23 or np.min(delta_q) <= -0.23:
            print('********************************** over joint velocity limit: ', delta_q)
            delta_q = delta_q / np.max(np.abs(delta_q)) * 0.23
            # delta_q = np.zeros((6, 1))
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def semi_track_depth(self):
        """
        Control FOV only consider the depth information
        :return:
        """
        # pass
        # calculate vision parameters(cVc/Errors) and publish the control signals
        self.actions = [0.0, 0.0, self.pred_action.z * self.depth_zz_speed_coef]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=self.tool_depth, flag=1)
        # self.vision_params_pub.publish(self.vision_params)
        cVc = self.vision_params.Parameter
        homo_delta = self.vision_params.ImageError[0:2]

        # drive the ur5 robot
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        if self.SettingGlobal and self.robot is not None:
            self.robot._init_global(self.joint_pos)
            self.SettingGlobal = False
        # print("step1: cVc: ", cVc)
        # print("homo_delta: ", homo_delta)
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True  # False means the target has already achieved
            self.Track_z_Mode = False
        # print('delta_q: ', delta_q)
        if np.max(delta_q) >= 0.23 or np.min(delta_q) <= -0.23:
            print('********************************** over joint velocity limit: ', delta_q)
            delta_q = delta_q / np.max(np.abs(delta_q)) * 0.23
            # delta_q = np.zeros((6, 1))
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def move_init(self):
        """
        Moving to the init camera position
        :return:
        """
        # drive the ur5 robot, avoid init for multiple times
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        if self.SettingGlobal and self.robot is not None:
            self.robot._init_global(self.joint_pos)
            self.SettingGlobal = False

        cVc = self.robot.bVc_to_cVc(self.joint_pos)
        homo_delta = np.array([0, 0]) / 2.0
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)

        print("----------------------------------------------bTc_init----move_init: \n", self.robot.init_bTc)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True  # False means the target has already achieved
            self.Return_Mode = False
            # self.Return_Track_Mode = False
            self.Return_Track_Finish_flag = True
        print('---------------------------------------------delta_q----move_init: ', delta_q)

        if np.max(delta_q) >= 0.23 or np.min(delta_q) <= -0.23:
            print('********************************** over joint velocity limit: ', delta_q)
            delta_q = delta_q / np.max(np.abs(delta_q)) * 0.23
            # delta_q = np.zeros((6, 1))

        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def zoom_in(self):
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        if self.SettingGlobal and self.robot is not None:
            self.robot._init_global(self.joint_pos)
            self.SettingGlobal = False

        cVc, self.zoomin_action_once_flag = self.robot.rVc_to_cVc(self.joint_pos, flag=self.zoomin_action_once_flag, dir=1, workspace=(self.Workspace_Inter-self.closest_dist)/1000)
        # print('zoomin_action_once_flag: ', self.zoomin_action_once_flag)
        if self.robot.out_constraint and cVc[2] > 0:
            cVc[2] = 0
        if self.robot.close_constraint and cVc[2] < 0:
            cVc[2] = 0
        homo_delta = np.array([0, 0]) / 2.0
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True  # False means the target has already achieved
            self.Zoomin_Mode = False

        if np.max(delta_q) >= 0.23 or np.min(delta_q) <= -0.23:
            print('********************************** over joint velocity limit: ', delta_q)
            delta_q = delta_q / np.max(np.abs(delta_q)) * 0.23
            # delta_q = np.zeros((6, 1))
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def zoom_out(self):
        cVc, self.zoomout_action_once_flag = self.robot.rVc_to_cVc(self.joint_pos, flag=self.zoomout_action_once_flag, dir=-1, workspace=(self.Workspace_Inter-self.closest_dist)/1000)
        # print('zoomout_action_once_flag: ', self.zoomout_action_once_flag)
        if self.robot.out_constraint and cVc[2] > 0:
            cVc[2] = 0
        if self.robot.close_constraint and cVc[2] < 0:
            cVc[2] = 0
        homo_delta = np.array([0, 0]) / 2.0
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True  # False means the target has already achieved
            self.Zoomout_Mode = False

        if np.max(delta_q) >= 0.23 or np.min(delta_q) <= -0.23:
            print('********************************** over joint velocity limit: ', delta_q)
            delta_q = delta_q / np.max(np.abs(delta_q)) * 0.23
            # delta_q = np.zeros((6, 1))
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def cal_vision_params(self, Kc, actions, depth, flag):
        if flag == 0:  # mannually control: constant depth
            c_z = depth
        elif flag == 1:  # automatic control: use the tool depth
            # print('depth: ', depth)
            if depth > 0.01 and depth < 0.20:
                # print('success :', depth)
                c_z = depth
            else:
                c_z = 0.1  # use the default depth when the depth is invalid
        k_image = 0.025  # velocity of movement 0.05
        homo_delta = np.array([actions[0], actions[1]]) / 2.0
        u_ = actions[0] * Kc[0, 0] / 2.0
        v_ = actions[1] * Kc[1, 1] / 2.0
        # Lmatrix = np.array([[-Kc[0, 0]/c_z, 0, u_/c_z], [0, -Kc[1, 1]/c_z, v_/c_z]])
        # uvn = np.dot(np.linalg.pinv(Kc), np.array([240, 320, 1.0]))
        # # print(uvn)
        # Lmatrix = np.array([[-1/c_z, 0, uvn[0]/c_z], [0, -1/c_z, uvn[1]/c_z]])
        Lmatrix = np.array([[-1 / c_z, 0, 0.0 / c_z], [0, -1 / c_z, 0.0 / c_z]]) # pre
        # Lmatrix = np.array([[- Kc[0, 0] / c_z, 0, 0.0 / c_z], [0, - Kc[1, 1] / c_z, 0.0 / c_z]]) # cur
        cVc = -1 * k_image * np.dot(np.linalg.pinv(Lmatrix), homo_delta)
        d_error = actions[2]
        # cVc[2] += -1 * k_image * 0.0015 * d_error
        cVc[2] += -1 * k_image * 0.10 * d_error * 3 # depth control -1 * k_image * 0.10 * d_error # pre # *3 add by 0424 bin
        # cVc[2] += -1 * k_image * d_error # depth control -1 * k_image * 0.10 * d_error # cur
        # print("cVc = \n", cVc)

        # set boundary of the position cmd
        step_limitation = 0.10
        if np.linalg.norm(cVc) > step_limitation:
            cVc = cVc * step_limitation / np.linalg.norm(cVc)
        # vision_params = ur5_vision_msg()
        vision_params = ur5_vision_msg
        vision_params.Parameter = cVc[0:3]  # new cmd
        vision_params.ImageError = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vision_params.ImageError[0:2] = homo_delta[0:2]  # publish img errors
        # vision_params.ImageError[2:4] = np.array([0.0, 0.0]) # publish target current positions
        # vision_params.ImageError[4:6] = h_dy[0:2] # publish desired img features
        # vision_params.ImageError[6] = 0.0 # d0 here
        # vision_params.ImageError[7] = d_error # publish desired img features
        return vision_params

        #  ----------- new gui design --------------

    def index2uterus_motion_left(self, index, img, input_1080=False):
        if index == 0:
            if self.uterus_angle_pitch > 64:
                img = paste_bbox_img(img, self.uterus_panel_up_bbox_left_1080p, self.uterus_panel_up_click_limit_icon_left_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_pitch > 61:
                img = paste_bbox_img(img, self.uterus_panel_up_bbox_left_1080p, self.uterus_panel_up_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_up_bbox_left_1080p, self.uterus_panel_up_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
            self.UterusUp_Mode = True
        elif index == 1:
            if self.uterus_angle_pitch < -19:
                img = paste_bbox_img(img, self.uterus_panel_down_bbox_left_1080p, self.uterus_panel_down_click_limit_icon_left_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_pitch < -16:
                img = paste_bbox_img(img, self.uterus_panel_down_bbox_left_1080p, self.uterus_panel_down_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_down_bbox_left_1080p, self.uterus_panel_down_click_icon_left_1080p,alpha=self.alpha_each_click_gui)
            self.UterusDown_Mode = True
        elif index == 2:
            if self.uterus_angle_yaw > 25:
                img = paste_bbox_img(img, self.uterus_panel_left_bbox_left_1080p, self.uterus_panel_left_click_limit_icon_left_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_yaw > 23:
                img = paste_bbox_img(img, self.uterus_panel_left_bbox_left_1080p, self.uterus_panel_left_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_left_bbox_left_1080p, self.uterus_panel_left_click_icon_left_1080p,alpha=self.alpha_each_click_gui)
            self.UterusLeft_Mode = True
        elif index == 3:
            if self.uterus_angle_yaw < -25:
                img = paste_bbox_img(img, self.uterus_panel_right_bbox_left_1080p, self.uterus_panel_right_click_limit_icon_left_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_yaw < -23:
                img = paste_bbox_img(img, self.uterus_panel_right_bbox_left_1080p, self.uterus_panel_right_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_right_bbox_left_1080p, self.uterus_panel_right_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
            self.UterusRight_Mode = True
        elif index == 4:
            if self.uterus_angle_insertion > 48:
                img = paste_bbox_img(img, self.uterus_panel_insert_bbox_left_1080p, self.uterus_panel_insert_click_limit_icon_left_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_insertion > 45:
                img = paste_bbox_img(img, self.uterus_panel_insert_bbox_left_1080p, self.uterus_panel_insert_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_insert_bbox_left_1080p, self.uterus_panel_insert_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
            self.UterusInsert_Mode = True
        elif index == 5:
            if self.uterus_angle_insertion < 2:
                img = paste_bbox_img(img, self.uterus_panel_retract_bbox_left_1080p, self.uterus_panel_retract_click_limit_icon_left_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_insertion < 5:
                img = paste_bbox_img(img, self.uterus_panel_retract_bbox_left_1080p, self.uterus_panel_retract_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_retract_bbox_left_1080p, self.uterus_panel_retract_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
            self.UterusRetract_Mode = True
        return img

    def index2uterus_motion_right(self, index, img, input_1080=False):
        if index == 0:
            if self.uterus_angle_pitch > 64:
                img = paste_bbox_img(img, self.uterus_panel_up_bbox_right_1080p, self.uterus_panel_up_click_limit_icon_right_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_pitch > 61:
                img = paste_bbox_img(img, self.uterus_panel_up_bbox_right_1080p, self.uterus_panel_up_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_up_bbox_right_1080p, self.uterus_panel_up_click_icon_right_1080p, alpha=self.alpha_each_click_gui)
            self.UterusUp_Mode = True
        elif index == 1:
            if self.uterus_angle_pitch < -19:
                img = paste_bbox_img(img, self.uterus_panel_down_bbox_right_1080p, self.uterus_panel_down_click_limit_icon_right_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_pitch < -16:
                img = paste_bbox_img(img, self.uterus_panel_down_bbox_right_1080p, self.uterus_panel_down_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_down_bbox_right_1080p, self.uterus_panel_down_click_icon_right_1080p, alpha=self.alpha_each_click_gui)
            self.UterusDown_Mode = True
        elif index == 2:
            if self.uterus_angle_yaw > 25:
                img = paste_bbox_img(img, self.uterus_panel_left_bbox_right_1080p, self.uterus_panel_left_click_limit_icon_right_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_yaw > 23:
                img = paste_bbox_img(img, self.uterus_panel_left_bbox_right_1080p, self.uterus_panel_left_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_left_bbox_right_1080p, self.uterus_panel_left_click_icon_right_1080p, alpha=self.alpha_each_click_gui)
            self.UterusLeft_Mode = True
        elif index == 3:
            if self.uterus_angle_yaw < -25:
                img = paste_bbox_img(img, self.uterus_panel_right_bbox_right_1080p, self.uterus_panel_right_click_limit_icon_right_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_yaw < -23:
                img = paste_bbox_img(img, self.uterus_panel_right_bbox_right_1080p, self.uterus_panel_right_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_right_bbox_right_1080p, self.uterus_panel_right_click_icon_right_1080p, alpha=self.alpha_each_click_gui)
            self.UterusRight_Mode = True
        elif index == 4:
            if self.uterus_angle_insertion > 48:
                img = paste_bbox_img(img, self.uterus_panel_insert_bbox_right_1080p, self.uterus_panel_insert_click_limit_icon_right_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_insertion > 45:
                img = paste_bbox_img(img, self.uterus_panel_insert_bbox_right_1080p, self.uterus_panel_insert_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_insert_bbox_right_1080p, self.uterus_panel_insert_click_icon_right_1080p, alpha=self.alpha_each_click_gui)
            self.UterusInsert_Mode = True
        elif index == 5:
            if self.uterus_angle_insertion < 2:
                img = paste_bbox_img(img, self.uterus_panel_retract_bbox_right_1080p, self.uterus_panel_retract_click_limit_icon_right_1080p, alpha=self.alpha_each_click_gui)
            elif self.uterus_angle_insertion < 5:
                img = paste_bbox_img(img, self.uterus_panel_retract_bbox_right_1080p, self.uterus_panel_retract_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_gui)
            else:
                img = paste_bbox_img(img, self.uterus_panel_retract_bbox_right_1080p, self.uterus_panel_retract_click_icon_right_1080p, alpha=self.alpha_each_click_gui)
            self.UterusRetract_Mode = True
        return img

    @debug_class('MainWindow')
    @pyqtSlot()
    def pushButton_end_clicked(self):
        """
        terminate the process
        :return:
        """
        self.video_flag = False

    @debug_class('MainWindow')
    def draw_img(self, img, actions):
        """
        draw the action results in the image
        :param img:
        :param actions:
        :return:
        """
        h, w, c = img.shape
        # print('w, h: ', w, h)
        rect_wh = 5
        # left/right action
        if actions[0] > 0:  # left case
            pt1 = (0, int((0.5 - actions[0] / 2.0) * h))
            pt2 = (rect_wh, int((0.5 + actions[0] / 2.0) * h))
        elif actions[0] < 0:  # right case
            pt1 = (w - rect_wh, int((0.5 + actions[0] / 2.0) * h))
            pt2 = (w, int((0.5 - actions[0] / 2.0) * h))
        else:
            pt1 = (0, 0)
            pt2 = (0, 0)

        # up/down action
        if actions[1] > 0:  # up case
            pt3 = (int((0.5 - actions[1] / 2.0) * w), 0)
            pt4 = (int((0.5 + actions[1] / 2.0) * w), rect_wh)
        elif actions[1] < 0:  # down case
            pt3 = (int((0.5 + actions[1] / 2.0) * w), h - rect_wh)
            pt4 = (int((0.5 - actions[1] / 2.0) * w), h)
        else:
            pt3 = (0, 0)
            pt4 = (0, 0)

        # draw the rect in image
        img = cv2.rectangle(img, pt1, pt2, (255, 255, 0), thickness=-1)
        img = cv2.rectangle(img, pt3, pt4, (255, 255, 0), thickness=-1)
        return img

    @debug_class('MainWindow')
    def draw_img_Refer_BBox(self, img, bbox, color, thickness=1, lineType=cv2.LINE_AA):
        """
        bbox: (cx, cy, w, h) are relative ratios
        """
        height, width = img.shape[:2]
        topleft_x = (bbox.cx - bbox.w / 2) * width
        topleft_y = (bbox.cy - bbox.h / 2) * height
        bottomright_x = (bbox.cx + bbox.w / 2) * width
        bottomright_y = (bbox.cy + bbox.h / 2) * height
        img = cv2.rectangle(img, (int(topleft_x), int(topleft_y)), (int(bottomright_x), int(bottomright_y)), color,
                            thickness=thickness, lineType=lineType)
        # print('refer bbox info: ', topleft_x, topleft_y, bottomright_x, bottomright_y)
        return img

    @debug_class('MainWindow')
    def draw_img_GUI_left_osfa(self, imgR_1080p, position, position_main):
        if (self.gui_display_time_delay < self.delay_main_keep_small) and self.mode == 'None':

            if is_multipos_in_bbox(position, self.main_menu_bbox_left_large):
                self.gui_main_panel_time[0, 1] = self.gui_main_panel_time[0, 2] = self.gui_main_panel_time[0, 3] = time.time()
                self.gui_main_panel_time_delay = time.time() - self.gui_main_panel_time
                self.gui_display_time = time.time()
                self.gui_display_time_delay = 0.0
                if self.gui_main_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                    self.mode = 'Menu'
                    self.menu_panel_time = np.ones((1,3)) * time.time()
                    self.menu_panel_time_delay = np.zeros((1, 3))
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p, self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p, self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_small_1080p, self.main_menu_icon_small_1080p, alpha=self.alpha_main_small_gui)

                self.gui_main_panel_time = np.ones((1,4)) * time.time()
                self.gui_main_panel_time_delay = np.zeros((1, 4))
                self.gui_display_time_delay = time.time() - self.gui_display_time

        elif (self.gui_display_time_delay >= self.delay_main_keep_small) and self.mode == 'None':
            self.gui_display = False

        else:
            self.gui_main_panel_time = np.ones((1, 4)) * time.time()
            self.gui_main_panel_time_delay = np.zeros((1, 4))

        if self.mode == 'Menu':
            if is_multipos_in_bbox(position, self.main_uterus_bbox_left_large):
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p,
                                            self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_left_large_1080p,
                                            self.main_uterus_left_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_left_large_1080p,
                                            self.main_camera_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, 1:] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            elif is_multipos_in_bbox(position, self.main_camera_bbox_left_large):
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p,
                                            self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_left_large_1080p,
                                            self.main_uterus_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_left_large_1080p,
                                            self.main_camera_left_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, 0] = time.time()
                self.menu_panel_time[0, 2] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time
            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p,
                                            self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_left_large_1080p,
                                            self.main_uterus_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_left_large_1080p,
                                            self.main_camera_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, :2] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            if self.menu_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                    self.mode = 'Uterus'
                    self.uterus_panel_time = np.ones((1, 7)) * time.time()
                    self.uterus_panel_time = np.ones((1, 7)) * time.time()
                    self.uterus_panel_time_delay = np.zeros((1, 7))
            elif self.menu_panel_time_delay[0, 1] > self.delay_main_large_to_panel:
                    self.mode = 'Camera'
                    self.camera_panel_time = np.ones((1, 6)) * time.time()
                    self.camera_panel_time_delay = np.zeros((1, 6))

            elif self.menu_panel_time_delay[0, 2] > self.delay_panel_each * 3:
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False

        # Uterus panel
        if self.mode == 'Uterus':
            if self.uterus_interface_mode == 'high' or self.uterus_interface_mode == 'wide':
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_panel_bbox_left_large_1080p, self.uterus_panel_click_icon_large_left_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_up_bbox_left_1080p, self.uterus_panel_up_icon_left_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_down_bbox_left_1080p, self.uterus_panel_down_icon_left_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_left_bbox_left_1080p, self.uterus_panel_left_icon_left_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_right_bbox_left_1080p, self.uterus_panel_right_icon_left_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_insert_bbox_left_1080p, self.uterus_panel_insert_icon_left_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_retract_bbox_left_1080p, self.uterus_panel_retract_icon_left_1080p, alpha=0.8)

                if is_multipos_in_bbox(position, self.uterus_panel_up_bbox_left):  # or is_pos_in_bbox(position_main, self.uterus_panel_up_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 0] = time.time() - self.uterus_panel_time[0, 0]
                else:
                    self.uterus_panel_time[0, 0] = time.time()
                    self.uterus_panel_time_delay[0, 0] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_down_bbox_left):  # or is_pos_in_bbox(position_main, self.uterus_panel_down_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 1] = time.time() - self.uterus_panel_time[0, 1]
                else:
                    self.uterus_panel_time[0, 1] = time.time()
                    self.uterus_panel_time_delay[0, 1] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_left_bbox_left):  # or is_pos_in_bbox(position_main, self.uterus_panel_left_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 2] = time.time() - self.uterus_panel_time[0, 2]
                else:
                    self.uterus_panel_time[0, 2] = time.time()
                    self.uterus_panel_time_delay[0, 2] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_right_bbox_left):  #  or is_pos_in_bbox(position_main, self.uterus_panel_right_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 3] = time.time() - self.uterus_panel_time[0, 3]
                else:
                    self.uterus_panel_time[0, 3] = time.time()
                    self.uterus_panel_time_delay[0, 3] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_insert_bbox_left):  #  or is_pos_in_bbox(position_main, self.uterus_panel_insert_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 4] = time.time() - self.uterus_panel_time[0, 4]
                else:
                    self.uterus_panel_time[0, 4] = time.time()
                    self.uterus_panel_time_delay[0, 4] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_retract_bbox_left):  #  or is_pos_in_bbox(position_main, self.uterus_panel_retract_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 5] = time.time() - self.uterus_panel_time[0, 5]
                else:
                    self.uterus_panel_time[0, 5] = time.time()
                    self.uterus_panel_time_delay[0, 5] = 0

                if (self.uterus_panel_time_delay[0, :6] == self.uterus_panel_time_delay_old[0, :6]).all():
                    self.uterus_panel_time[0, :6] = time.time()
                    self.uterus_panel_time_delay[0, 6] = time.time() - self.uterus_panel_time[0, 6]

                uterus_panel_time_delay_sort = np.sort(self.uterus_panel_time_delay[0, :-1])
                if self.uterus_panel_time_delay[0, 6] > self.delay_main_keep_small:
                    self.uterus_panel_time[0, :] = time.time()
                    self.uterus_panel_time_delay[0, :] = 0.0
                    self.mode = 'None'
                    self.gui_display = False
                elif uterus_panel_time_delay_sort[-1] < self.delay_panel_uterus:
                    pass
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] < self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-1])
                    imgR_1080p = self.index2uterus_motion_left(select_index[1], imgR_1080p, input_1080=True)
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] > self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-2])
                    imgR_1080p = self.index2uterus_motion_left(select_index[1], imgR_1080p, input_1080=True)
                self.uterus_panel_time_delay_old = copy.deepcopy(self.uterus_panel_time_delay)

        # Lock the main tool as we want
        if self.mode == 'Dominant':
            imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_panel_bbox_left_large_1080p, self.camera_panel_click_icon_large_left_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomin_bbox_left_1080p, self.camera_panel_zoomin_icon_left_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomout_bbox_left_1080p, self.camera_panel_zoomout_icon_left_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_setzero_bbox_left_1080p, self.camera_panel_setzero_icon_left_1080p, alpha=0.7)

            self.main_main_tool_img_1080p = self.main_domain_tool_click_ls_1080p_left[self.main_tool_id]
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomauto_bbox_left_1080p, self.main_main_tool_img_1080p, alpha=0.7)

            # TODO: deal with the position now and get the main tool index
            self.dominant_time_delay = time.time() - self.dominant_time
            if self.dominant_time_delay > self.delay_main_keep_small*2:
                self.mode = 'None'
                self.gui_display = False
            else:
                interval_x = 6.0  # need < 8 parts
                interval_y = 4.0
                # interval_x + 1
                # interval_y + 1
                part = 60  # totally 16 parts, origin value: 60 here

                x1y1x2y2 = [int(interval_x * part), int(interval_y * part),
                            self.main_panel_img_W - int(interval_x * part),
                            self.main_panel_img_H - int(interval_y * part)]
                x1y1x2y2_1080p = [int(x1y1x2y2[0] * self.main_3d_img_W / self.main_panel_img_W),
                                  int(x1y1x2y2[1] * self.main_3d_img_H / self.main_panel_img_H),
                                  int(x1y1x2y2[2] * self.main_3d_img_W / self.main_panel_img_W),
                                  int(x1y1x2y2[3] * self.main_3d_img_H / self.main_panel_img_H)]

                cv2.rectangle(imgR_1080p, (x1y1x2y2_1080p[0], x1y1x2y2_1080p[1]),
                              (x1y1x2y2_1080p[2], x1y1x2y2_1080p[3]), (255, 255, 255), thickness=1,
                              lineType=cv2.LINE_AA)

                exist, idx = is_multipos_in_bbox_with_idx(position, x1y1x2y2)
                if exist:
                    self.dominant_tool_time_delay = time.time() - self.dominant_tool_time
                    if self.dominant_tool_time_delay > self.delay_panel_dorminant:  # control the exchange time count_threshold * 2
                        self.main_tool_id = self.aux_tool_index[idx]
                        self.dominant_tool_time = time.time()
                        self.dominant_tool_time_delay = 0
                else:
                    self.dominant_tool_time = time.time()
                    self.dominant_tool_time_delay = 0

        # Camera panel
        if self.mode == 'Camera':
            imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_panel_bbox_left_large_1080p, self.camera_panel_click_icon_large_left_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomin_bbox_left_1080p, self.camera_panel_zoomin_icon_left_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomout_bbox_left_1080p, self.camera_panel_zoomout_icon_left_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_setzero_bbox_left_1080p, self.camera_panel_setzero_icon_left_1080p, alpha=0.7)

            self.main_main_tool_img_1080p = self.main_domain_tool_ls_1080p_left[self.main_tool_id]
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomauto_bbox_left_1080p, self.main_main_tool_img_1080p, alpha=0.7)

            # camera_panel_H = round(3*resize_H/4/4)
            if is_multipos_in_bbox(position, self.camera_panel_zoomin_bbox_left):
                self.camera_panel_time[0, 1:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.camera_panel_zoomout_bbox_left):
                self.camera_panel_time[0, 0] = time.time()
                self.camera_panel_time[0, 2:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.camera_panel_zoomauto_bbox_left):
                self.camera_panel_time[0, :2] = time.time()
                self.camera_panel_time[0, 3:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.camera_panel_setzero_bbox_left):
                self.camera_panel_time[0, :3] = time.time()
                self.camera_panel_time[0, 4:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.main_camera_bbox_left_large):
                self.camera_panel_time[0, :4] = time.time()
                self.camera_panel_time[0, 5:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            else:
                self.virtual_button_flag = False
                self.camera_panel_time[0, :5] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time

            if self.camera_panel_time_delay[0, 0] > self.delay_panel_each:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomin_bbox_left_1080p,
                                            self.camera_panel_zoomin_click_icon_left_1080p, alpha=0.9)

                # self.Return_Track_Mode = False  # only adjust the depth
                # self.Zoomin_Mode = True
                # self.zoomin_action_once_flag = True
                if self.virtual_button_flag is False:
                    self.Zoomin_FIX_Mode = True
                    self.virtual_button_flag = True
                    # self.vis_setting_depth_time = time.time()
            elif self.camera_panel_time_delay[0, 1] > self.delay_panel_each:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomout_bbox_left_1080p,
                                            self.camera_panel_zoomout_click_icon_left_1080p, alpha=0.9)

                # self.Return_Track_Mode = False  # only adjust the depth
                # self.Zoomout_Mode = True
                # self.zoomout_action_once_flag = True
                if self.virtual_button_flag is False:
                    self.Zoomout_FIX_Mode = True
                    self.virtual_button_flag = True
            elif self.camera_panel_time_delay[0, 2] > self.delay_panel_each:
                # self.main_main_tool_img_1080p = self.main_domain_tool_click_ls_1080p_left[self.main_tool_id]
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomauto_bbox_left_1080p, self.main_main_tool_img_1080p, alpha=0.7)

                # self.Return_Track_Mode = False  # only adjust the depth
                # self.Track_z_Mode = True
                self.mode = 'Dominant'
                self.dominant_tool_time = time.time()
                self.dominant_tool_time_delay = 0
                self.dominant_time = time.time()
                self.diminant_time_delay = 0

            elif self.camera_panel_time_delay[0, 3] > self.delay_panel_each:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_setzero_bbox_left_1080p,
                                            self.camera_panel_setzero_click_icon_left_1080p, alpha=0.9)

                # self.Return_Mode = True
                self.target_status = False  # necessary: to be used to back to initial position
                self.Return_Track_Mode = True
            elif self.camera_panel_time_delay[0, 5] > self.delay_panel_each * 3:
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False

        # Setting panel
        if self.mode == 'Setting':  # or self.Setting_Mode:
            if self.SettingLap0:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_0_icon_left_1080p, alpha=0.8)
            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_30_icon_left_1080p, alpha=0.8)

            # if self.SettingHandLeft:
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_left_icon_left_1080p, alpha=0.8)
            # 
            # else:
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_right_icon_left_1080p, alpha=0.8)

            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p, self.setting_panel_uspeed_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p, self.setting_panel_cspeed_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p, self.setting_panel_box_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p, self.setting_panel_depth_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_rotation_bbox_left_1080p, self.setting_panel_rotation_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_tip_usage_bbox_left_1080p, self.setting_panel_tip_usage_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_global_init_icon_left_1080p, alpha=0.8)

            if is_multipos_in_bbox(position, self.setting_panel_lap_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_hand_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_uspeed_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_cspeed_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_box_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_depth_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_rotation_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_tip_usage_bbox_left):
                self.setting_panel_count[0, 8] = 0
            else:
                self.setting_panel_count[0, 8] = self.setting_panel_count[0, 8] + 1
                self.setting_panel_count[0, 0] = self.setting_panel_count[0, 1] = self.setting_panel_count[0, 2] = \
                self.setting_panel_count[0, 3] = self.setting_panel_count[0, 4] = self.setting_panel_count[0, 5] = \
                self.setting_panel_count[0, 6] = self.setting_panel_count[0, 7] = 0

            if self.setting_panel_count[0, 0] > self.delay_panel_each or int(self.setting_button_index) == 1:
                self.SettingLap = True
                if self.SettingLap0:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_0_click_icon_left_1080p, alpha=0.9)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_30_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 1] > self.delay_panel_each or int(self.setting_button_index) == 2:
                self.SettingHand = True
                # if self.SettingHandLeft:
                #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_left_click_icon_left_1080p, alpha=0.9)
                # else:
                #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_right_click_icon_left_1080p, alpha=0.9)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_global_init_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 2] > self.delay_panel_each or int(self.setting_button_index) == 3:
                self.SettingUspeed = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p, self.setting_panel_uspeed_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 3] > self.delay_panel_each or int(self.setting_button_index) == 4:
                self.SettingCspeed = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p, self.setting_panel_cspeed_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 4] > self.delay_panel_each or int(self.setting_button_index) == 5:
                self.SettingBox = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p, self.setting_panel_box_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 5] > self.delay_panel_each or int(self.setting_button_index) == 6:
                self.SettingDepth = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p, self.setting_panel_depth_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 6] > self.delay_panel_each or int(self.setting_button_index) == 7:
                self.SettingRotation = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_rotation_bbox_left_1080p, self.setting_panel_rotation_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 7] > self.delay_panel_each or int(self.setting_button_index) == 8:
                self.SettingTipUsage = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_tip_usage_bbox_left_1080p, self.setting_panel_tip_usage_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 8] > self.delay_main_keep_small or (self.Setting_flag is False):
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False
                self.mode_trigger_tool = False

            paste_text_img(imgR_1080p, f'x {self.SettingUspeedValue:.1f}', self.setting_panel_uspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'x {self.SettingCspeedValue:.1f}', self.setting_panel_cspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'{int((2.0 - self.SettingBoxValue) * 50)}' + ' %', self.setting_panel_box_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img_long(imgR_1080p, f'x {self.SettingDepthRate:.1f}' + ' (' + str(int(self.mid_range)) + ' mm)', self.setting_panel_depth_bbox_left_1080p, self.setting_font_1080p)
            paste_text_img(imgR_1080p, f'{self.SettingRotationValue:.1f}' + ' deg', self.setting_panel_rotation_bbox_left_1080p, self.setting_font_1080p)
            paste_text_img(imgR_1080p, str(self.SettingTipUsage0), self.setting_panel_tip_usage_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, str(self.SettingGlobal0), self.setting_panel_hand_bbox_left_1080p, self.setting_font_large_1080p)

        return imgR_1080p

    @debug_class('MainWindow')
    def draw_img_GUI_right_osfa(self, right_img_1080p, position, position_main):
        imgR_1080p = right_img_1080p.copy()
        if (self.gui_display_time_delay < self.delay_main_keep_small) and self.mode == 'None':

            if is_multipos_in_bbox(position, self.main_menu_bbox_right_large):
                self.gui_main_panel_time[0, 1] = self.gui_main_panel_time[0, 2] = self.gui_main_panel_time[0, 3] = time.time()
                self.gui_main_panel_time_delay = time.time() - self.gui_main_panel_time
                self.gui_display_time = time.time()
                self.gui_display_time_delay = 0.0
                if self.gui_main_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                    self.mode = 'Menu'
                    self.menu_panel_time = np.ones((1,3)) * time.time()
                    self.menu_panel_time_delay = np.zeros((1, 3))
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p, self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p, self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_small_1080p, self.main_menu_icon_small_1080p, alpha=self.alpha_main_small_gui)

                self.gui_main_panel_time = np.ones((1,4)) * time.time()
                self.gui_main_panel_time_delay = np.zeros((1, 4))
                self.gui_display_time_delay = time.time() - self.gui_display_time

        elif (self.gui_display_time_delay >= self.delay_main_keep_small) and self.mode == 'None':
            self.gui_display = False

        else:
            self.gui_main_panel_time = np.ones((1, 4)) * time.time()
            self.gui_main_panel_time_delay = np.zeros((1, 4))

        if self.mode == 'Menu':
            if is_multipos_in_bbox(position, self.main_uterus_bbox_right_large):
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p,
                                            self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_right_large_1080p,
                                            self.main_uterus_right_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_right_large_1080p,
                                            self.main_camera_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, 1:] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            elif is_multipos_in_bbox(position, self.main_camera_bbox_right_large):
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p,
                                            self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_right_large_1080p,
                                            self.main_uterus_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_right_large_1080p,
                                            self.main_camera_right_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, 0] = time.time()
                self.menu_panel_time[0, 2] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time
            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p,
                                            self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_right_large_1080p,
                                            self.main_uterus_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_right_large_1080p,
                                            self.main_camera_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, :2] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            if self.menu_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                    self.mode = 'Uterus'
                    self.uterus_panel_time = np.ones((1, 7)) * time.time()
                    self.uterus_panel_time = np.ones((1, 7)) * time.time()
                    self.uterus_panel_time_delay = np.zeros((1, 7))
            elif self.menu_panel_time_delay[0, 1] > self.delay_main_large_to_panel:
                    self.mode = 'Camera'
                    self.camera_panel_time = np.ones((1, 6)) * time.time()
                    self.camera_panel_time_delay = np.zeros((1, 6))

            elif self.menu_panel_time_delay[0, 2] > self.delay_panel_each * 3:
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False

        # Uterus panel
        if self.mode == 'Uterus':
            imgR_1080p = right_img_1080p.copy()
            if self.uterus_interface_mode == 'high' or self.uterus_interface_mode == 'wide':
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_panel_bbox_right_large_1080p, self.uterus_panel_click_icon_large_right_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_up_bbox_right_1080p, self.uterus_panel_up_icon_right_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_down_bbox_right_1080p, self.uterus_panel_down_icon_right_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_left_bbox_right_1080p, self.uterus_panel_left_icon_right_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_right_bbox_right_1080p, self.uterus_panel_right_icon_right_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_insert_bbox_right_1080p, self.uterus_panel_insert_icon_right_1080p, alpha=0.8)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_retract_bbox_right_1080p, self.uterus_panel_retract_icon_right_1080p, alpha=0.8)

                if is_multipos_in_bbox(position, self.uterus_panel_up_bbox_right):  #  or is_pos_in_bbox(position_main, self.uterus_panel_up_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 0] = time.time() - self.uterus_panel_time[0, 0]
                else:
                    self.uterus_panel_time[0, 0] = time.time()
                    self.uterus_panel_time_delay[0, 0] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_down_bbox_right):  #  or is_pos_in_bbox(position_main, self.uterus_panel_down_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 1] = time.time() - self.uterus_panel_time[0, 1]
                else:
                    self.uterus_panel_time[0, 1] = time.time()
                    self.uterus_panel_time_delay[0, 1] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_left_bbox_right):  #  or is_pos_in_bbox(position_main, self.uterus_panel_left_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 2] = time.time() - self.uterus_panel_time[0, 2]
                else:
                    self.uterus_panel_time[0, 2] = time.time()
                    self.uterus_panel_time_delay[0, 2] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_right_bbox_right):  #  or is_pos_in_bbox(position_main, self.uterus_panel_right_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 3] = time.time() - self.uterus_panel_time[0, 3]
                else:
                    self.uterus_panel_time[0, 3] = time.time()
                    self.uterus_panel_time_delay[0, 3] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_insert_bbox_right):  #  or is_pos_in_bbox(position_main, self.uterus_panel_insert_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 4] = time.time() - self.uterus_panel_time[0, 4]
                else:
                    self.uterus_panel_time[0, 4] = time.time()
                    self.uterus_panel_time_delay[0, 4] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_retract_bbox_right):  #  or is_pos_in_bbox(position_main, self.uterus_panel_retract_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 5] = time.time() - self.uterus_panel_time[0, 5]
                else:
                    self.uterus_panel_time[0, 5] = time.time()
                    self.uterus_panel_time_delay[0, 5] = 0

                if (self.uterus_panel_time_delay[0, :6] == self.uterus_panel_time_delay_old[0, :6]).all():
                    self.uterus_panel_time[0, :6] = time.time()
                    self.uterus_panel_time_delay[0, 6] = time.time() - self.uterus_panel_time[0, 6]

                uterus_panel_time_delay_sort = np.sort(self.uterus_panel_time_delay[0, :-1])
                if self.uterus_panel_time_delay[0, 6] > self.delay_main_keep_small:
                    imgR_1080p = imgR_1080p.copy()
                    self.uterus_panel_time[0, :] = time.time()
                    self.uterus_panel_time_delay[0, :] = 0.0
                    self.mode = 'None'
                    self.gui_display = False
                elif uterus_panel_time_delay_sort[-1] < self.delay_panel_uterus:
                    pass
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] < self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-1])
                    imgR_1080p = self.index2uterus_motion_right(select_index[1], imgR_1080p, input_1080=True)
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] > self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-2])
                    imgR_1080p = self.index2uterus_motion_right(select_index[1], imgR_1080p, input_1080=True)
                self.uterus_panel_time_delay_old = copy.deepcopy(self.uterus_panel_time_delay)

        # Lock the main tool as we want
        if self.mode == 'Dominant':
            imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_panel_bbox_right_large_1080p, self.camera_panel_click_icon_large_right_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomin_bbox_right_1080p, self.camera_panel_zoomin_icon_right_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomout_bbox_right_1080p, self.camera_panel_zoomout_icon_right_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_setzero_bbox_right_1080p, self.camera_panel_setzero_icon_right_1080p, alpha=0.7)

            self.main_main_tool_img_1080p = self.main_domain_tool_click_ls_1080p_right[self.main_tool_id]
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomauto_bbox_right_1080p, self.main_main_tool_img_1080p, alpha=0.7)

            # TODO: deal with the position now and get the main tool index
            self.dominant_time_delay = time.time() - self.dominant_time
            if self.dominant_time_delay > self.delay_main_keep_small*2:
                self.mode = 'None'
                self.gui_display = False
            else:
                interval_x = 6.0  # need < 8 parts
                interval_y = 4.0
                # interval_x + 1
                # interval_y + 1
                part = 60  # totally 16 parts, origin value: 60 here

                x1y1x2y2 = [int(interval_x * part), int(interval_y * part),
                            self.main_panel_img_W - int(interval_x * part),
                            self.main_panel_img_H - int(interval_y * part)]
                x1y1x2y2_1080p = [int(x1y1x2y2[0] * self.main_3d_img_W / self.main_panel_img_W),
                                  int(x1y1x2y2[1] * self.main_3d_img_H / self.main_panel_img_H),
                                  int(x1y1x2y2[2] * self.main_3d_img_W / self.main_panel_img_W),
                                  int(x1y1x2y2[3] * self.main_3d_img_H / self.main_panel_img_H)]

                cv2.rectangle(imgR_1080p, (x1y1x2y2_1080p[0], x1y1x2y2_1080p[1]),
                              (x1y1x2y2_1080p[2], x1y1x2y2_1080p[3]), (255, 255, 255), thickness=1,
                              lineType=cv2.LINE_AA)

                exist, idx = is_multipos_in_bbox_with_idx(position, x1y1x2y2)
                if exist:
                    self.dominant_tool_time_delay = time.time() - self.dominant_tool_time
                    if self.dominant_tool_time_delay > self.delay_panel_dorminant:
                        self.main_tool_id = self.aux_tool_index[idx]
                        self.dominant_tool_time = time.time()
                        self.dominant_tool_time_delay = 0
                else:
                    self.dominant_tool_time = time.time()
                    self.dominant_tool_time_delay = 0

        # Camera panel
        if self.mode == 'Camera':
            imgR_1080p = right_img_1080p.copy()

            imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_panel_bbox_right_large_1080p, self.camera_panel_click_icon_large_right_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomin_bbox_right_1080p, self.camera_panel_zoomin_icon_right_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomout_bbox_right_1080p, self.camera_panel_zoomout_icon_right_1080p, alpha=0.7)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_setzero_bbox_right_1080p, self.camera_panel_setzero_icon_right_1080p, alpha=0.7)

            self.main_main_tool_img_1080p = self.main_domain_tool_ls_1080p_right[self.main_tool_id]
            imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomauto_bbox_right_1080p, self.main_main_tool_img_1080p, alpha=0.7)

            # camera_panel_H = round(3*resize_H/4/4)
            if is_multipos_in_bbox(position, self.camera_panel_zoomin_bbox_right):
                self.camera_panel_time[0, 1:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.camera_panel_zoomout_bbox_right):
                self.camera_panel_time[0, 0] = time.time()
                self.camera_panel_time[0, 2:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.camera_panel_zoomauto_bbox_right):
                self.camera_panel_time[0, :2] = time.time()
                self.camera_panel_time[0, 3:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.camera_panel_setzero_bbox_right):
                self.camera_panel_time[0, :3] = time.time()
                self.camera_panel_time[0, 4:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            elif is_multipos_in_bbox(position, self.main_camera_bbox_right_large):
                self.camera_panel_time[0, :4] = time.time()
                self.camera_panel_time[0, 5:] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time
            else:
                self.virtual_button_flag = False
                self.camera_panel_time[0, :5] = time.time()
                self.camera_panel_time_delay = time.time() - self.camera_panel_time

            if self.camera_panel_time_delay[0, 0] > self.delay_panel_each:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomin_bbox_right_1080p,
                                            self.camera_panel_zoomin_click_icon_right_1080p, alpha=0.9)

                # self.Return_Track_Mode = False  # only adjust the depth
                # self.Zoomin_Mode = True
                # self.zoomin_action_once_flag = True
                if self.virtual_button_flag is False:
                    self.Zoomin_FIX_Mode = True
                    self.virtual_button_flag = True
                    # self.vis_setting_depth_time = time.time()
            elif self.camera_panel_time_delay[0, 1] > self.delay_panel_each:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_zoomout_bbox_right_1080p,
                                            self.camera_panel_zoomout_click_icon_right_1080p, alpha=0.9)

                # self.Return_Track_Mode = False  # only adjust the depth
                # self.Zoomout_Mode = True
                # self.zoomout_action_once_flag = True
                if self.virtual_button_flag is False:
                    self.Zoomout_FIX_Mode = True
                    self.virtual_button_flag = True
            elif self.camera_panel_time_delay[0, 2] > self.delay_panel_each:
                self.mode = 'Dominant'
                self.dominant_tool_time = time.time()
                self.dominant_tool_time_delay = 0
                self.dominant_time = time.time()
                self.diminant_time_delay = 0

            elif self.camera_panel_time_delay[0, 3] > self.delay_panel_each:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.camera_panel_setzero_bbox_right_1080p,
                                            self.camera_panel_setzero_click_icon_right_1080p, alpha=0.9)

                # self.Return_Mode = True
                self.target_status = False  # necessary: to be used to back to initial position
                self.Return_Track_Mode = True
            elif self.camera_panel_time_delay[0, 5] > self.delay_panel_each * 3:
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False

        # Setting panel
        if self.mode == 'Setting':  # or self.Setting_Mode:
            if self.SettingLap0:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_0_icon_left_1080p, alpha=0.8)
            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_30_icon_left_1080p, alpha=0.8)

            # if self.SettingHandLeft:
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_left_icon_left_1080p, alpha=0.8)
            #
            # else:
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_right_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_global_init_icon_left_1080p, alpha=0.8)

            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p, self.setting_panel_uspeed_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p, self.setting_panel_cspeed_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p, self.setting_panel_box_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p, self.setting_panel_depth_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_rotation_bbox_left_1080p, self.setting_panel_rotation_icon_left_1080p, alpha=0.8)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_tip_usage_bbox_left_1080p, self.setting_panel_tip_usage_icon_left_1080p, alpha=0.8)

            if is_multipos_in_bbox(position, self.setting_panel_lap_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_hand_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_uspeed_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_cspeed_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_box_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_depth_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_rotation_bbox_left):
                self.setting_panel_count[0, 8] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_tip_usage_bbox_left):
                self.setting_panel_count[0, 8] = 0
            else:
                self.setting_panel_count[0, 8] = self.setting_panel_count[0, 8] + 1
                self.setting_panel_count[0, 0] = self.setting_panel_count[0, 1] = self.setting_panel_count[0, 2] = \
                self.setting_panel_count[0, 3] = self.setting_panel_count[0, 4] = self.setting_panel_count[0, 5] = \
                self.setting_panel_count[0, 6] = self.setting_panel_count[0, 7] = 0

            if self.setting_panel_count[0, 0] > self.delay_panel_each or int(self.setting_button_index) == 1:
                self.SettingLap = True
                if self.SettingLap0:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_0_click_icon_left_1080p, alpha=0.9)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_panel_lap_30_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 1] > self.delay_panel_each or int(self.setting_button_index) == 2:
                self.SettingHand = True
                # if self.SettingHandLeft:
                #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_left_click_icon_left_1080p, alpha=0.9)
                # else:
                #     imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_hand_right_click_icon_left_1080p, alpha=0.9)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_panel_global_init_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 2] > self.delay_panel_each or int(self.setting_button_index) == 3:
                self.SettingUspeed = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p, self.setting_panel_uspeed_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 3] > self.delay_panel_each or int(self.setting_button_index) == 4:
                self.SettingCspeed = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p, self.setting_panel_cspeed_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 4] > self.delay_panel_each or int(self.setting_button_index) == 5:
                self.SettingBox = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p, self.setting_panel_box_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 5] > self.delay_panel_each or int(self.setting_button_index) == 6:
                self.SettingDepth = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p, self.setting_panel_depth_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 6] > self.delay_panel_each or int(self.setting_button_index) == 7:
                self.SettingRotation = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_rotation_bbox_left_1080p, self.setting_panel_rotation_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 7] > self.delay_panel_each or int(self.setting_button_index) == 8:
                self.SettingTipUsage = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_tip_usage_bbox_left_1080p, self.setting_panel_tip_usage_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 8] > self.delay_main_keep_small or (self.Setting_flag is False):
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False
                self.mode_trigger_tool = False

            paste_text_img(imgR_1080p, f'x {self.SettingUspeedValue:.1f}', self.setting_panel_uspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'x {self.SettingCspeedValue:.1f}', self.setting_panel_cspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'{int((2.0 - self.SettingBoxValue) * 50)}' + ' %', self.setting_panel_box_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img_long(imgR_1080p, f'x {self.SettingDepthRate:.1f}' + ' (' + str(int(self.mid_range)) + ' mm)', self.setting_panel_depth_bbox_left_1080p, self.setting_font_1080p)
            paste_text_img(imgR_1080p, f'{self.SettingRotationValue:.1f}' + ' deg', self.setting_panel_rotation_bbox_left_1080p, self.setting_font_1080p)
            paste_text_img(imgR_1080p, str(self.SettingTipUsage0), self.setting_panel_tip_usage_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, str(self.SettingGlobal0), self.setting_panel_hand_bbox_left_1080p, self.setting_font_large_1080p)

        return imgR_1080p

    @debug_class('MainWindow')
    def draw_img_GUI_left_pure(self, imgR_1080p, position, position_main):
        if (self.gui_display_time_delay < self.delay_main_keep_small) and self.mode == 'None':

            if is_multipos_in_bbox(position, self.main_menu_bbox_left_large) or \
                    is_pos_in_bbox(position_main, self.main_menu_bbox_left_large):
                self.gui_main_panel_time[0, 1] = self.gui_main_panel_time[0, 2] = self.gui_main_panel_time[0, 3] = time.time()
                self.gui_main_panel_time_delay = time.time() - self.gui_main_panel_time
                self.gui_display_time = time.time()
                self.gui_display_time_delay = 0.0
                if self.gui_main_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                    self.mode = 'Menu'
                    self.menu_panel_time = np.ones((1,3)) * time.time()
                    self.menu_panel_time_delay = np.zeros((1, 3))
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p, self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p, self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_small_1080p, self.main_menu_icon_small_1080p, alpha=self.alpha_main_small_gui)

                self.gui_main_panel_time = np.ones((1,4)) * time.time()
                self.gui_main_panel_time_delay = np.zeros((1, 4))
                self.gui_display_time_delay = time.time() - self.gui_display_time

        elif (self.gui_display_time_delay >= self.delay_main_keep_small) and self.mode == 'None':
            self.gui_display = False

        else:
            self.gui_main_panel_time = np.ones((1, 4)) * time.time()
            self.gui_main_panel_time_delay = np.zeros((1, 4))

        if self.mode == 'Menu':
            if is_multipos_in_bbox(position, self.main_uterus_bbox_left_large) or \
                    is_pos_in_bbox(position_main, self.main_uterus_bbox_left_large):
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p,
                                            self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_left_large_1080p,
                                            self.main_uterus_left_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_left_large_1080p,
                #                             self.main_camera_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, 1:] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            # elif is_multipos_in_bbox(position, self.main_camera_bbox_left_large):
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p,
            #                                 self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_left_large_1080p,
            #                                 self.main_uterus_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
            #     # imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_left_large_1080p,
            #     #                             self.main_camera_left_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
            #     self.menu_panel_time[0, 0] = time.time()
            #     self.menu_panel_time[0, 2] = time.time()
            #     self.menu_panel_time_delay = time.time() - self.menu_panel_time
            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_left_large_1080p,
                                            self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_left_large_1080p,
                                            self.main_uterus_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_left_large_1080p,
                #                             self.main_camera_left_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, :2] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            if self.menu_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                    self.mode = 'Uterus'
                    self.uterus_panel_time = np.ones((1, 7)) * time.time()
                    self.uterus_panel_time = np.ones((1, 7)) * time.time()
                    self.uterus_panel_time_delay = np.zeros((1, 7))
            elif self.menu_panel_time_delay[0, 2] > self.delay_panel_each * 3:
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False

        # Uterus panel
        if self.mode == 'Uterus':
            if self.uterus_interface_mode == 'high' or self.uterus_interface_mode == 'wide':
                time_uterus_0 = time.time()
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_panel_bbox_left_large_1080p, self.uterus_panel_click_icon_large_left_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_up_bbox_left_1080p, self.uterus_panel_up_icon_left_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_down_bbox_left_1080p, self.uterus_panel_down_icon_left_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_left_bbox_left_1080p, self.uterus_panel_left_icon_left_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_right_bbox_left_1080p, self.uterus_panel_right_icon_left_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_insert_bbox_left_1080p, self.uterus_panel_insert_icon_left_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_retract_bbox_left_1080p, self.uterus_panel_retract_icon_left_1080p, alpha=self.alpha_each_gui)

                if is_multipos_in_bbox(position, self.uterus_panel_up_bbox_left) or is_pos_in_bbox(position_main, self.uterus_panel_up_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 0] = time.time() - self.uterus_panel_time[0, 0]
                else:
                    self.uterus_panel_time[0, 0] = time.time()
                    self.uterus_panel_time_delay[0, 0] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_down_bbox_left) or is_pos_in_bbox(position_main, self.uterus_panel_down_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 1] = time.time() - self.uterus_panel_time[0, 1]
                else:
                    self.uterus_panel_time[0, 1] = time.time()
                    self.uterus_panel_time_delay[0, 1] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_left_bbox_left) or is_pos_in_bbox(position_main, self.uterus_panel_left_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 2] = time.time() - self.uterus_panel_time[0, 2]
                else:
                    self.uterus_panel_time[0, 2] = time.time()
                    self.uterus_panel_time_delay[0, 2] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_right_bbox_left) or is_pos_in_bbox(position_main, self.uterus_panel_right_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 3] = time.time() - self.uterus_panel_time[0, 3]
                else:
                    self.uterus_panel_time[0, 3] = time.time()
                    self.uterus_panel_time_delay[0, 3] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_insert_bbox_left) or is_pos_in_bbox(position_main, self.uterus_panel_insert_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 4] = time.time() - self.uterus_panel_time[0, 4]
                else:
                    self.uterus_panel_time[0, 4] = time.time()
                    self.uterus_panel_time_delay[0, 4] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_retract_bbox_left) or is_pos_in_bbox(position_main, self.uterus_panel_retract_bbox_left):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 5] = time.time() - self.uterus_panel_time[0, 5]
                else:
                    self.uterus_panel_time[0, 5] = time.time()
                    self.uterus_panel_time_delay[0, 5] = 0

                if (self.uterus_panel_time_delay[0, :6] == self.uterus_panel_time_delay_old[0, :6]).all():
                    self.uterus_panel_time[0, :6] = time.time()
                    self.uterus_panel_time_delay[0, 6] = time.time() - self.uterus_panel_time[0, 6]

                uterus_panel_time_delay_sort = np.sort(self.uterus_panel_time_delay[0, :-1])
                if self.uterus_panel_time_delay[0, 6] > self.delay_main_keep_small:
                    self.uterus_panel_time[0, :] = time.time()
                    self.uterus_panel_time_delay[0, :] = 0.0
                    self.mode = 'None'
                    self.gui_display = False
                elif uterus_panel_time_delay_sort[-1] < self.delay_panel_uterus:
                    pass
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] < self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-1])
                    imgR_1080p = self.index2uterus_motion_left(select_index[1], imgR_1080p, input_1080=True)
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] > self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-2])
                    imgR_1080p = self.index2uterus_motion_left(select_index[1], imgR_1080p, input_1080=True)
                self.uterus_panel_time_delay_old = copy.deepcopy(self.uterus_panel_time_delay)
                time_uterus_1 = time.time()
                # print('^^^^^^^^^^^^^^^^^ time _uterus ^^^^^', time_uterus_1 - time_uterus_0)

        # Setting panel
        if self.mode == 'Setting':  # or self.Setting_Mode:
            if self.SettingLap0:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_uterus_panel_lap_0_icon_left_1080p, alpha=self.alpha_each_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_uterus_panel_lap_30_icon_left_1080p, alpha=self.alpha_each_gui)

            if self.SettingHandLeft:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_hand_left_icon_left_1080p, alpha=self.alpha_each_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_hand_right_icon_left_1080p, alpha=self.alpha_each_gui)
            # imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_global_init_icon_left_1080p, alpha=self.alpha_each_gui)

            time_setting_1 = time.time()
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p, self.setting_uterus_panel_uspeed_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p, self.setting_panel_cspeed_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p, self.setting_panel_rotation_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p, self.setting_uterus_panel_tip_usage_icon_left_1080p, alpha=self.alpha_each_gui)

            time_setting_done = time.time()

            # TODO: 20211224, avoid the puzzy bug
            # if is_multipos_in_bbox(position, self.setting_panel_lap_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_hand_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_uspeed_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_cspeed_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_box_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_depth_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_rotation_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # else:
            #     self.setting_panel_count[0, 7] = self.setting_panel_count[0, 7] + 1
            #     self.setting_panel_count[0, 0] = self.setting_panel_count[0, 1] = self.setting_panel_count[0, 2] = \
            #     self.setting_panel_count[0, 3] = self.setting_panel_count[0, 4] = self.setting_panel_count[0, 5] = \
            #     self.setting_panel_count[0, 6] = 0

            self.setting_panel_count[0, 7] = self.setting_panel_count[0, 7] + 1
            self.setting_panel_count[0, 0] = self.setting_panel_count[0, 1] = self.setting_panel_count[0, 2] = \
            self.setting_panel_count[0, 3] = self.setting_panel_count[0, 4] = self.setting_panel_count[0, 5] = \
            self.setting_panel_count[0, 6] = 0

            if self.setting_panel_count[0, 0] > self.delay_panel_each or int(self.setting_button_index) == 1:
                self.SettingLap = True
                if self.SettingLap0:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p,
                                                self.setting_uterus_panel_lap_0_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p,
                                                self.setting_uterus_panel_lap_30_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

            elif self.setting_panel_count[0, 1] > self.delay_panel_each or int(self.setting_button_index) == 2:
                self.SettingHand = True
                if self.SettingHandLeft:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p,
                                                self.setting_uterus_panel_hand_left_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p,
                                                self.setting_uterus_panel_hand_right_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, 
                #                             self.setting_uterus_panel_global_init_click_icon_left_1080p, alpha=self.alpha_each_gui)


            elif self.setting_panel_count[0, 2] > self.delay_panel_each or int(self.setting_button_index) == 3:
                self.SettingUspeed = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p,
                                            self.setting_uterus_panel_uspeed_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

            elif self.setting_panel_count[0, 3] > self.delay_panel_each or int(self.setting_button_index) == 4:
                self.SettingTipUsage = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p,
                                            self.setting_panel_cspeed_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 4] > self.delay_panel_each or int(self.setting_button_index) == 5:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p,
                                            self.setting_panel_rotation_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 5] > self.delay_panel_each or int(self.setting_button_index) == 6:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p,
                                            self.setting_uterus_panel_tip_usage_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 7] > self.delay_main_keep_small or (self.Setting_flag is False):
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False
                self.mode_trigger_tool = False

            paste_text_img(imgR_1080p, f'x {self.SettingUspeedValue:.1f}', self.setting_panel_uspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'x {self.SettingCspeedValue:.1f}', self.setting_panel_cspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'{self.SettingRotationValue:.1f}' + ' deg', self.setting_panel_box_bbox_left_1080p, self.setting_font_1080p)
            paste_text_img(imgR_1080p, str(self.SettingTipUsage0), self.setting_panel_depth_bbox_left_1080p, self.setting_font_large_1080p)
            # paste_text_img(imgR_1080p, str(self.SettingGlobal0), self.setting_panel_hand_bbox_left_1080p, self.setting_font_large_1080p)

        return imgR_1080p

    @debug_class('MainWindow')
    def draw_img_GUI_right_pure(self, imgR_1080p, position, position_main):
        if (self.gui_display_time_delay < self.delay_main_keep_small) and self.mode == 'None':

            if is_multipos_in_bbox(position, self.main_menu_bbox_right_large) or \
                    is_pos_in_bbox(position_main, self.main_menu_bbox_right_large):
                self.gui_main_panel_time[0, 1] = self.gui_main_panel_time[0, 2] = self.gui_main_panel_time[0, 3] = time.time()
                self.gui_main_panel_time_delay = time.time() - self.gui_main_panel_time
                self.gui_display_time = time.time()
                self.gui_display_time_delay = 0.0
                if self.gui_main_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                    self.mode = 'Menu'
                    self.menu_panel_time = np.ones((1,3)) * time.time()
                    self.menu_panel_time_delay = np.zeros((1, 3))
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p, self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p, self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_small_1080p, self.main_menu_icon_small_1080p, alpha=self.alpha_main_small_gui)

                self.gui_main_panel_time = np.ones((1,4)) * time.time()
                self.gui_main_panel_time_delay = np.zeros((1, 4))
                self.gui_display_time_delay = time.time() - self.gui_display_time

        elif (self.gui_display_time_delay >= self.delay_main_keep_small) and self.mode == 'None':
            self.gui_display = False

        else:
            self.gui_main_panel_time = np.ones((1, 4)) * time.time()
            self.gui_main_panel_time_delay = np.zeros((1, 4))

        if self.mode == 'Menu':
            if is_multipos_in_bbox(position, self.main_uterus_bbox_right_large) or \
                    is_pos_in_bbox(position_main, self.main_uterus_bbox_right_large):
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p,
                                            self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_right_large_1080p,
                                            self.main_uterus_right_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_right_large_1080p,
                #                             self.main_camera_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, 1:] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            # elif is_multipos_in_bbox(position, self.main_camera_bbox_right_large):
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p,
            #                                 self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
            #     imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_right_large_1080p,
            #                                 self.main_uterus_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
            #     # imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_right_large_1080p,
            #     #                             self.main_camera_right_icon_click_large_1080p, alpha=self.alpha_main_large_gui)
            #     self.menu_panel_time[0, 0] = time.time()
            #     self.menu_panel_time[0, 2] = time.time()
            #     self.menu_panel_time_delay = time.time() - self.menu_panel_time

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_menu_bbox_right_large_1080p,
                                            self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_bbox_right_large_1080p,
                                            self.main_uterus_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.main_camera_bbox_right_large_1080p,
                #                             self.main_camera_right_icon_large_1080p, alpha=self.alpha_main_large_gui)
                self.menu_panel_time[0, :2] = time.time()
                self.menu_panel_time_delay = time.time() - self.menu_panel_time

            if self.menu_panel_time_delay[0, 0] > self.delay_main_large_to_panel:
                self.mode = 'Uterus'
                self.uterus_panel_time = np.ones((1, 7)) * time.time()
                self.uterus_panel_time = np.ones((1, 7)) * time.time()
                self.uterus_panel_time_delay = np.zeros((1, 7))

            elif self.menu_panel_time_delay[0, 2] > self.delay_panel_each * 3:
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False

        # Uterus panel
        if self.mode == 'Uterus':
            if self.uterus_interface_mode == 'high' or self.uterus_interface_mode == 'wide':
                time_uterus_0 = time.time()
                imgR_1080p = paste_bbox_img(imgR_1080p, self.main_uterus_panel_bbox_right_large_1080p, self.uterus_panel_click_icon_large_right_1080p, alpha=self.alpha_main_large_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_up_bbox_right_1080p, self.uterus_panel_up_icon_right_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_down_bbox_right_1080p, self.uterus_panel_down_icon_right_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_left_bbox_right_1080p, self.uterus_panel_left_icon_right_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_right_bbox_right_1080p, self.uterus_panel_right_icon_right_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_insert_bbox_right_1080p, self.uterus_panel_insert_icon_right_1080p, alpha=self.alpha_each_gui)
                imgR_1080p = paste_bbox_img(imgR_1080p, self.uterus_panel_retract_bbox_right_1080p, self.uterus_panel_retract_icon_right_1080p, alpha=self.alpha_each_gui)

                if is_multipos_in_bbox(position, self.uterus_panel_up_bbox_right) or is_pos_in_bbox(position_main, self.uterus_panel_up_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 0] = time.time() - self.uterus_panel_time[0, 0]
                else:
                    self.uterus_panel_time[0, 0] = time.time()
                    self.uterus_panel_time_delay[0, 0] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_down_bbox_right) or is_pos_in_bbox(position_main, self.uterus_panel_down_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 1] = time.time() - self.uterus_panel_time[0, 1]
                else:
                    self.uterus_panel_time[0, 1] = time.time()
                    self.uterus_panel_time_delay[0, 1] = 0

                if is_multipos_in_bbox(position, self.uterus_panel_left_bbox_right) or is_pos_in_bbox(position_main, self.uterus_panel_left_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 2] = time.time() - self.uterus_panel_time[0, 2]
                else:
                    self.uterus_panel_time[0, 2] = time.time()
                    self.uterus_panel_time_delay[0, 2] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_right_bbox_right) or is_pos_in_bbox(position_main, self.uterus_panel_right_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 3] = time.time() - self.uterus_panel_time[0, 3]
                else:
                    self.uterus_panel_time[0, 3] = time.time()
                    self.uterus_panel_time_delay[0, 3] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_insert_bbox_right) or is_pos_in_bbox(position_main, self.uterus_panel_insert_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 4] = time.time() - self.uterus_panel_time[0, 4]
                else:
                    self.uterus_panel_time[0, 4] = time.time()
                    self.uterus_panel_time_delay[0, 4] = 0
                if is_multipos_in_bbox(position, self.uterus_panel_retract_bbox_right) or is_pos_in_bbox(position_main, self.uterus_panel_retract_bbox_right):
                    self.uterus_panel_time[0, 6] = time.time()
                    self.uterus_panel_time_delay[0, 5] = time.time() - self.uterus_panel_time[0, 5]
                else:
                    self.uterus_panel_time[0, 5] = time.time()
                    self.uterus_panel_time_delay[0, 5] = 0

                if (self.uterus_panel_time_delay[0, :6] == self.uterus_panel_time_delay_old[0, :6]).all():
                    self.uterus_panel_time[0, :6] = time.time()
                    self.uterus_panel_time_delay[0, 6] = time.time() - self.uterus_panel_time[0, 6]

                uterus_panel_time_delay_sort = np.sort(self.uterus_panel_time_delay[0, :-1])
                if self.uterus_panel_time_delay[0, 6] > self.delay_main_keep_small:
                    self.uterus_panel_time[0, :] = time.time()
                    self.uterus_panel_time_delay[0, :] = 0.0
                    self.mode = 'None'
                    self.gui_display = False
                elif uterus_panel_time_delay_sort[-1] < self.delay_panel_uterus:
                    pass
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] < self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-1])
                    imgR_1080p = self.index2uterus_motion_right(select_index[1], imgR_1080p, input_1080=True)
                elif uterus_panel_time_delay_sort[-1] > self.delay_panel_uterus and uterus_panel_time_delay_sort[-2] > self.delay_panel_uterus:
                    select_index = np.where(self.uterus_panel_time_delay == uterus_panel_time_delay_sort[-2])
                    imgR_1080p = self.index2uterus_motion_right(select_index[1], imgR_1080p, input_1080=True)
                self.uterus_panel_time_delay_old = copy.deepcopy(self.uterus_panel_time_delay)
                time_uterus_1 = time.time()
                # print('^^^^^^^^^^^^^^^^^ time _uterus ^^^^^', time_uterus_1 - time_uterus_0)

        # Setting panel
        if self.mode == 'Setting':  # or self.Setting_Mode:
            if self.SettingLap0:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_uterus_panel_lap_0_icon_left_1080p, alpha=self.alpha_each_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_uterus_panel_lap_30_icon_left_1080p, alpha=self.alpha_each_gui)

            if self.SettingHandLeft:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_hand_left_icon_left_1080p, alpha=self.alpha_each_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_hand_right_icon_left_1080p, alpha=self.alpha_each_gui)
            # imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_global_init_icon_left_1080p, alpha=self.alpha_each_gui)

            time_setting_1 = time.time()
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p, self.setting_uterus_panel_uspeed_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p, self.setting_panel_cspeed_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p, self.setting_panel_rotation_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p, self.setting_uterus_panel_tip_usage_icon_left_1080p, alpha=self.alpha_each_gui)

            time_setting_done = time.time()

            # TODO: 20211224, avoid the puzzy bug
            # if is_multipos_in_bbox(position, self.setting_panel_lap_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_hand_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_uspeed_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_cspeed_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_box_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_depth_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # elif is_multipos_in_bbox(position, self.setting_panel_rotation_bbox_left):
            #     self.setting_panel_count[0, 7] = 0
            # else:
            #     self.setting_panel_count[0, 7] = self.setting_panel_count[0, 7] + 1
            #     self.setting_panel_count[0, 0] = self.setting_panel_count[0, 1] = self.setting_panel_count[0, 2] = \
            #     self.setting_panel_count[0, 3] = self.setting_panel_count[0, 4] = self.setting_panel_count[0, 5] = \
            #     self.setting_panel_count[0, 6] = 0

            self.setting_panel_count[0, 7] = self.setting_panel_count[0, 7] + 1
            self.setting_panel_count[0, 0] = self.setting_panel_count[0, 1] = self.setting_panel_count[0, 2] = \
            self.setting_panel_count[0, 3] = self.setting_panel_count[0, 4] = self.setting_panel_count[0, 5] = \
            self.setting_panel_count[0, 6] = 0

            if self.setting_panel_count[0, 0] > self.delay_panel_each or int(self.setting_button_index) == 1:
                self.SettingLap = True
                if self.SettingLap0:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p,
                                                self.setting_uterus_panel_lap_0_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p,
                                                self.setting_uterus_panel_lap_30_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

            elif self.setting_panel_count[0, 1] > self.delay_panel_each or int(self.setting_button_index) == 2:
                self.SettingHand = True
                if self.SettingHandLeft:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p,
                                                self.setting_uterus_panel_hand_left_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p,
                                                self.setting_uterus_panel_hand_right_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, 
                #                             self.setting_uterus_panel_global_init_click_icon_left_1080p, alpha=self.alpha_each_gui)


            elif self.setting_panel_count[0, 2] > self.delay_panel_each or int(self.setting_button_index) == 3:
                self.SettingUspeed = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p,
                                            self.setting_uterus_panel_uspeed_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

            elif self.setting_panel_count[0, 3] > self.delay_panel_each or int(self.setting_button_index) == 4:
                self.SettingTipUsage = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p,
                                            self.setting_panel_cspeed_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 4] > self.delay_panel_each or int(self.setting_button_index) == 5:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p,
                                            self.setting_panel_rotation_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 5] > self.delay_panel_each or int(self.setting_button_index) == 6:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p,
                                            self.setting_uterus_panel_tip_usage_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 7] > self.delay_main_keep_small or (self.Setting_flag is False):
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False
                self.mode_trigger_tool = False

            paste_text_img(imgR_1080p, f'x {self.SettingUspeedValue:.1f}', self.setting_panel_uspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'x {self.SettingCspeedValue:.1f}', self.setting_panel_cspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'{self.SettingRotationValue:.1f}' + ' deg', self.setting_panel_box_bbox_left_1080p, self.setting_font_1080p)
            paste_text_img(imgR_1080p, str(self.SettingTipUsage0), self.setting_panel_depth_bbox_left_1080p, self.setting_font_large_1080p)
            # paste_text_img(imgR_1080p, str(self.SettingGlobal0), self.setting_panel_hand_bbox_left_1080p, self.setting_font_large_1080p)

        return imgR_1080p

    @debug_class('MainWindow')
    def draw_img_GUI_left_none(self, imgR_1080p, position):
        # Setting panel
        if self.mode == 'Setting':  # or self.Setting_Mode:
            if self.SettingLap0:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_uterus_panel_lap_0_icon_left_1080p, alpha=self.alpha_each_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p, self.setting_uterus_panel_lap_30_icon_left_1080p, alpha=self.alpha_each_gui)

            if self.SettingHandLeft:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_hand_left_icon_left_1080p, alpha=self.alpha_each_gui)

            else:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_hand_right_icon_left_1080p, alpha=self.alpha_each_gui)
            # imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, self.setting_uterus_panel_global_init_icon_left_1080p, alpha=self.alpha_each_gui)

            time_setting_1 = time.time()
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p, self.setting_uterus_panel_uspeed_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p, self.setting_panel_cspeed_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p, self.setting_panel_rotation_icon_left_1080p, alpha=self.alpha_each_gui)
            imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p, self.setting_uterus_panel_tip_usage_icon_left_1080p, alpha=self.alpha_each_gui)

            time_setting_done = time.time()

            # TODO: 20211224, avoid the puzzy bug
            if is_multipos_in_bbox(position, self.setting_panel_lap_bbox_left):
                self.setting_panel_count[0, 7] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_hand_bbox_left):
                self.setting_panel_count[0, 7] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_uspeed_bbox_left):
                self.setting_panel_count[0, 7] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_cspeed_bbox_left):
                self.setting_panel_count[0, 7] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_box_bbox_left):
                self.setting_panel_count[0, 7] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_depth_bbox_left):
                self.setting_panel_count[0, 7] = 0
            elif is_multipos_in_bbox(position, self.setting_panel_rotation_bbox_left):
                self.setting_panel_count[0, 7] = 0
            else:
                self.setting_panel_count[0, 7] = self.setting_panel_count[0, 7] + 1
                self.setting_panel_count[0, 0] = self.setting_panel_count[0, 1] = self.setting_panel_count[0, 2] = \
                self.setting_panel_count[0, 3] = self.setting_panel_count[0, 4] = self.setting_panel_count[0, 5] = \
                self.setting_panel_count[0, 6] = 0

            if self.setting_panel_count[0, 0] > self.delay_panel_each or int(self.setting_button_index) == 1:
                self.SettingLap = True
                if self.SettingLap0:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p,
                                                self.setting_uterus_panel_lap_0_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_lap_bbox_left_1080p,
                                                self.setting_uterus_panel_lap_30_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

            elif self.setting_panel_count[0, 1] > self.delay_panel_each or int(self.setting_button_index) == 2:
                self.SettingHand = True
                if self.SettingHandLeft:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p,
                                                self.setting_uterus_panel_hand_left_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
                else:
                    imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p,
                                                self.setting_uterus_panel_hand_right_click_icon_left_1080p, alpha=self.alpha_each_click_gui)
                # imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_hand_bbox_left_1080p, 
                #                             self.setting_uterus_panel_global_init_click_icon_left_1080p, alpha=self.alpha_each_gui)


            elif self.setting_panel_count[0, 2] > self.delay_panel_each or int(self.setting_button_index) == 3:
                self.SettingUspeed = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_uspeed_bbox_left_1080p,
                                            self.setting_uterus_panel_uspeed_click_icon_left_1080p, alpha=self.alpha_each_click_gui)

            elif self.setting_panel_count[0, 3] > self.delay_panel_each or int(self.setting_button_index) == 4:
                self.SettingTipUsage = True
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_cspeed_bbox_left_1080p,
                                            self.setting_panel_cspeed_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 4] > self.delay_panel_each or int(self.setting_button_index) == 5:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_box_bbox_left_1080p,
                                            self.setting_panel_rotation_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 5] > self.delay_panel_each or int(self.setting_button_index) == 6:
                imgR_1080p = paste_bbox_img(imgR_1080p, self.setting_panel_depth_bbox_left_1080p,
                                            self.setting_uterus_panel_tip_usage_click_icon_left_1080p, alpha=0.9)

            elif self.setting_panel_count[0, 7] > self.delay_main_keep_small or (self.Setting_flag is False):
                imgR_1080p = imgR_1080p.copy()
                self.mode = 'None'
                self.gui_display = False
                self.mode_trigger_tool = False

            paste_text_img(imgR_1080p, f'x {self.SettingUspeedValue:.1f}', self.setting_panel_uspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'x {self.SettingCspeedValue:.1f}', self.setting_panel_cspeed_bbox_left_1080p, self.setting_font_large_1080p)
            paste_text_img(imgR_1080p, f'{self.SettingRotationValue:.1f}' + ' deg', self.setting_panel_box_bbox_left_1080p, self.setting_font_1080p)
            paste_text_img(imgR_1080p, str(self.SettingTipUsage0), self.setting_panel_depth_bbox_left_1080p, self.setting_font_large_1080p)
            # paste_text_img(imgR_1080p, str(self.SettingGlobal0), self.setting_panel_hand_bbox_left_1080p, self.setting_font_large_1080p)

        return imgR_1080p

    def load_panel_icons(self, W, H, scale=1.0):
        print('load_icons for panel')
        resize_W = W
        resize_H = H
        resize_scale = scale

        # region ------ main panel buttons ------
        part_W = round(resize_W / 16 * resize_scale)
        part_W_l = round(resize_W / 16 * 1.5 * resize_scale)
        part_H = round(resize_H / 5)

        self.edge_bbox_topleft = [0, 0, 2 * part_W, part_H]
        self.edge_bbox_topright = [resize_W - 2 * part_W, 0, resize_W, part_H]

        # Main panel bbox left
        self.main_menu_bbox_left_small = [0, 0, part_W_l, part_H]
        self.main_menu_bbox_left_large = [0, 0, 2 * part_W, part_H]
        self.main_uterus_bbox_left_large = [0, 1 * part_H, 2 * part_W, 2 * part_H]
        self.main_camera_bbox_left_large = [0, 2 * part_H, 2 * part_W, 3 * part_H]
        # main panel bbox right
        self.main_menu_bbox_right_small = [resize_W - part_W_l, 0, resize_W, part_H]
        self.main_menu_bbox_right_large = [resize_W - 2 * part_W, 0, resize_W, part_H]
        self.main_uterus_bbox_right_large = [resize_W - 2 * part_W, 1 * part_H, resize_W, 2 * part_H]
        self.main_camera_bbox_right_large = [resize_W - 2 * part_W, 2 * part_H, resize_W, 3 * part_H]

        # ----- main tool ----------------
        resize_W_tool = round(1.6 * part_W)
        resize_H_tool = round(0.6 * part_H)
        main_domain_tool1_icon = cv2.imread("./sources/gui_fig_v3/main/tool1m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool2_icon = cv2.imread("./sources/gui_fig_v3/main/tool2m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool3_icon = cv2.imread("./sources/gui_fig_v3/main/tool3m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool4_icon = cv2.imread("./sources/gui_fig_v3/main/tool4m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool5_icon = cv2.imread("./sources/gui_fig_v3/main/tool5m.png", cv2.IMREAD_UNCHANGED)
        self.main_domain_tool1_icon = cv2.resize(main_domain_tool1_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool2_icon = cv2.resize(main_domain_tool2_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool3_icon = cv2.resize(main_domain_tool3_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool4_icon = cv2.resize(main_domain_tool4_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool5_icon = cv2.resize(main_domain_tool5_icon, (resize_W_tool, resize_H_tool))

        self.main_domain_tool1_icon_bg = add_bg([236,236,236], self.main_domain_tool1_icon)
        self.main_domain_tool2_icon_bg = add_bg([236,236,236], self.main_domain_tool2_icon)
        self.main_domain_tool3_icon_bg = add_bg([236,236,236], self.main_domain_tool3_icon)
        self.main_domain_tool4_icon_bg = add_bg([236,236,236], self.main_domain_tool4_icon)
        self.main_domain_tool5_icon_bg = add_bg([236,236,236], self.main_domain_tool5_icon)

        self.main_domain_tool_ls = [self.main_domain_tool1_icon_bg, self.main_domain_tool2_icon_bg,
                                    self.main_domain_tool3_icon_bg, self.main_domain_tool4_icon_bg,
                                    self.main_domain_tool5_icon_bg]
        print('load main_domain_tool1_icon')

        main_point_icon = cv2.imread('./sources/gui_fig_v3/main/color_03.png', cv2.IMREAD_UNCHANGED)
        aux_point_icon = cv2.imread('./sources/gui_fig_v3/main/main_point_icon_4.png', cv2.IMREAD_UNCHANGED)
        self.main_point_icon = cv2.resize(main_point_icon, (32, 32))
        self.aux_point_icon = cv2.resize(aux_point_icon, (32, 32))
        #endregion

        # region ------ camera interface buttons ------
        camera_panel_W = round(resize_W / 8 * resize_scale)
        camera_panel_H = round(resize_H / 5)

        self.main_camera_panel_bbox_left_large = [0, 2 * part_H, 2 * part_W, 3 * part_H]
        self.main_camera_panel_bbox_right_large = [resize_W - 2 * part_W, 2 * part_H, resize_W, 3 * part_H]

        self.camera_panel_zoomin_bbox_left = [0, 0, camera_panel_W, camera_panel_H]
        self.camera_panel_zoomout_bbox_left = [0, camera_panel_H, camera_panel_W, 2 * camera_panel_H]
        self.camera_panel_zoomauto_bbox_left = [0, 3 * camera_panel_H, camera_panel_W, 4 * camera_panel_H]
        self.camera_panel_setzero_bbox_left = [0, 4 * camera_panel_H, camera_panel_W, 5 * camera_panel_H]

        self.camera_panel_zoomin_bbox_right = [resize_W - camera_panel_W, 0, resize_W, camera_panel_H]
        self.camera_panel_zoomout_bbox_right = [resize_W - camera_panel_W, camera_panel_H, resize_W, 2 * camera_panel_H]
        self.camera_panel_zoomauto_bbox_right = [resize_W - camera_panel_W, 3 * camera_panel_H, resize_W, 4 * camera_panel_H]
        self.camera_panel_setzero_bbox_right = [resize_W - camera_panel_W, 4 * camera_panel_H, resize_W, 5 * camera_panel_H]
        #endregion

        # region  ------ uterus interface buttons ------
        if self.uterus_interface_mode == 'high':
            uterus_panel_W = round(resize_W / 6.5 * resize_scale)
            uterus_panel_H = round(resize_H / 20)

            self.main_uterus_panel_bbox_left_large = [0, 0, uterus_panel_W, uterus_panel_H*2]
            self.main_uterus_panel_bbox_right_large = [resize_W - uterus_panel_W, 0, resize_W, uterus_panel_H*2]

            self.uterus_panel_up_bbox_left = [0, 2 * uterus_panel_H, uterus_panel_W, 5 * uterus_panel_H]
            self.uterus_panel_down_bbox_left = [0, 5 * uterus_panel_H, uterus_panel_W, 8 * uterus_panel_H]
            self.uterus_panel_left_bbox_left = [0, 8 * uterus_panel_H, uterus_panel_W, 11 * uterus_panel_H]
            self.uterus_panel_right_bbox_left = [0, 11 * uterus_panel_H, uterus_panel_W, 14 * uterus_panel_H]
            self.uterus_panel_insert_bbox_left = [0, 14 * uterus_panel_H, uterus_panel_W, 17 * uterus_panel_H]
            self.uterus_panel_retract_bbox_left = [0, 17 * uterus_panel_H, uterus_panel_W, 20 * uterus_panel_H]

            self.uterus_panel_up_bbox_right = [resize_W - uterus_panel_W, 2 * uterus_panel_H, resize_W, 5 * uterus_panel_H]
            self.uterus_panel_down_bbox_right = [resize_W - uterus_panel_W, 5 * uterus_panel_H, resize_W, 8 * uterus_panel_H]
            self.uterus_panel_left_bbox_right = [resize_W - uterus_panel_W, 8 * uterus_panel_H, resize_W,  11 * uterus_panel_H]
            self.uterus_panel_right_bbox_right = [resize_W - uterus_panel_W, 11 * uterus_panel_H, resize_W, 14 * uterus_panel_H]
            self.uterus_panel_insert_bbox_right = [resize_W - uterus_panel_W, 14 * uterus_panel_H, resize_W, 17 * uterus_panel_H]
            self.uterus_panel_retract_bbox_right = [resize_W - uterus_panel_W, 17 * uterus_panel_H, resize_W, 20 * uterus_panel_H]

        print('load_uterus_icon_success')
        # endregion

        # region  ------ setting interface buttons ------
        # ------------ test_setting -----------
        setting_panel_W = round(resize_W / 8 * resize_scale)
        setting_panel_H = round(resize_H / 8)

        self.setting_panel_lap_bbox_left = [0, 0, setting_panel_W, setting_panel_H]
        self.setting_panel_hand_bbox_left = [0, setting_panel_H, setting_panel_W, 2 * setting_panel_H]
        self.setting_panel_uspeed_bbox_left = [0, 2 * setting_panel_H, setting_panel_W, 3 * setting_panel_H]
        self.setting_panel_cspeed_bbox_left = [0, 3 * setting_panel_H, setting_panel_W, 4 * setting_panel_H]
        self.setting_panel_box_bbox_left = [0, 4 * setting_panel_H, setting_panel_W, 5 * setting_panel_H]
        self.setting_panel_depth_bbox_left = [0, 5 * setting_panel_H, setting_panel_W, 6 * setting_panel_H]
        self.setting_panel_rotation_bbox_left = [0, 6 * setting_panel_H, setting_panel_W, 7 * setting_panel_H]
        self.setting_panel_tip_usage_bbox_left = [0, 7 * setting_panel_H, setting_panel_W, 8 * setting_panel_H]

        self.setting_panel_lap_bbox_right = [resize_W - setting_panel_W, 0, resize_W, setting_panel_H]
        self.setting_panel_hand_bbox_right = [resize_W - setting_panel_W, setting_panel_H, resize_W, 2 * setting_panel_H]
        self.setting_panel_uspeed_bbox_right = [resize_W - setting_panel_W, 2 * setting_panel_H, resize_W, 3 * setting_panel_H]
        self.setting_panel_cspeed_bbox_right = [resize_W - setting_panel_W, 3 * setting_panel_H, resize_W, 4 * setting_panel_H]
        self.setting_panel_box_bbox_right = [resize_W - setting_panel_W, 4 * setting_panel_H, resize_W, 5 * setting_panel_H]
        self.setting_panel_depth_bbox_right = [resize_W - setting_panel_W, 5 * setting_panel_H, resize_W, 6 * setting_panel_H]
        self.setting_panel_rotation_bbox_right = [resize_W - setting_panel_W, 6 * setting_panel_H, resize_W, 7 * setting_panel_H]
        self.setting_panel_tip_usage_bbox_right = [resize_W - setting_panel_W, 7 * setting_panel_H, resize_W, 8 * setting_panel_H]
        self.setting_font = resize_W * 5.5555e-4
        # endregion

        # region  ------ panel count ------
        self.dorminant_tool_count = 0  # one number for counting the tool to be setup as the main tool
        self.setting_panel_count = np.zeros((1, 9))  # extra one for disappear counting

        ####################################### panel time-based count
        # self.main_panel_time = np.zeros((1, 5))  # extra one for disappear counting
        # self.dorminant_tool_count = 0  # one number for counting the tool to be setup as the main tool
        # self.setting_panel_count = np.zeros((1, 8))  # extra one for disappear counting
        # endregion

        ####################################### gui mode
        self.mode = 'None'
        self.gui_display = False
        # self.gui_display_count = 0
        # print('self.mode = None')

        print('load_icons for panel 1080p done ')

    def load_1080p_icons(self, W=1920, H=1080, scale=1.0):
        print('load_icons for panel 1080p')
        resize_W = W
        resize_H = H
        resize_scale = scale

        part_W = round(resize_W / 16 * resize_scale)
        part_W_l = round(resize_W / 16 * 1.5 * resize_scale)
        part_H = round(resize_H / 5)

        # region  ------ main interface buttons ------
        main_icon_small_bg = cv2.imread("./sources/gui_fig_v3/main_v2/main_small_bg.png", cv2.IMREAD_UNCHANGED)
        main_icon_large_left_bg = cv2.imread('./sources/gui_fig_v3/main_v2/main_large_bg_left.png', cv2.IMREAD_UNCHANGED)
        main_icon_large_right_bg = cv2.imread('./sources/gui_fig_v3/main_v2/main_large_bg_right.png', cv2.IMREAD_UNCHANGED)
        self.main_icon_small_bg_1080p = cv2.resize(main_icon_small_bg, (part_W_l, part_H))
        self.main_icon_large_left_bg_1080p = cv2.resize(main_icon_large_left_bg, (part_W * 2, part_H))
        self.main_icon_large_right_bg_1080p = cv2.resize(main_icon_large_right_bg, (part_W * 2, part_H))
        main_left_icon_click_large_bg = cv2.imread("./sources/gui_fig_v3/main_v2/main_large_click_bg_left.png", cv2.IMREAD_UNCHANGED)
        main_right_icon_click_large_bg = cv2.imread("./sources/gui_fig_v3/main_v2/main_large_click_bg_right.png", cv2.IMREAD_UNCHANGED)
        self.main_left_icon_click_large_bg_1080p = cv2.resize(main_left_icon_click_large_bg, (part_W * 2, part_H))
        self.main_right_icon_click_large_bg_1080p = cv2.resize(main_right_icon_click_large_bg, (part_W * 2, part_H))

        main_menu_icon_small = cv2.imread('./sources/gui_fig_v3/main_v2/menu_small.png', cv2.IMREAD_UNCHANGED)
        main_menu_left_icon_large = cv2.imread('./sources/gui_fig_v3/main_v2/menu_large_left.png', cv2.IMREAD_UNCHANGED)
        main_menu_right_icon_large = cv2.imread('./sources/gui_fig_v3/main_v2/menu_large_right.png', cv2.IMREAD_UNCHANGED)
        self.main_menu_icon_small_1080p = cv2.resize(main_menu_icon_small, (part_W_l, part_H))
        self.main_menu_left_icon_large_1080p = cv2.resize(main_menu_left_icon_large, (part_W * 2, part_H))
        self.main_menu_right_icon_large_1080p = cv2.resize(main_menu_right_icon_large, (part_W * 2, part_H))

        main_uterus_left_icon_large = cv2.imread("./sources/gui_fig_v3/main_v2/uterus_large_left.png", cv2.IMREAD_UNCHANGED)
        main_uterus_right_icon_large = cv2.imread("./sources/gui_fig_v3/main_v2/uterus_large_right.png", cv2.IMREAD_UNCHANGED)
        self.main_uterus_left_icon_large_1080p = cv2.resize(main_uterus_left_icon_large, (part_W * 2, part_H))
        self.main_uterus_right_icon_large_1080p = cv2.resize(main_uterus_right_icon_large, (part_W * 2, part_H))

        main_camera_left_icon_large = cv2.imread("./sources/gui_fig_v3/main_v2/camera_large_left.png", cv2.IMREAD_UNCHANGED)
        main_camera_right_icon_large = cv2.imread("./sources/gui_fig_v3/main_v2/camera_large_right.png", cv2.IMREAD_UNCHANGED)
        self.main_camera_left_icon_large_1080p = cv2.resize(main_camera_left_icon_large, (part_W * 2, part_H))
        self.main_camera_right_icon_large_1080p = cv2.resize(main_camera_right_icon_large, (part_W * 2, part_H))

        main_uterus_left_icon_click_large = cv2.imread("./sources/gui_fig_v3/main_v2/uterus_large_left.png", cv2.IMREAD_UNCHANGED)
        main_uterus_right_icon_click_large = cv2.imread("./sources/gui_fig_v3/main_v2/uterus_large_right.png", cv2.IMREAD_UNCHANGED)
        self.main_uterus_left_icon_click_large_1080p = cv2.resize(main_uterus_left_icon_click_large, (part_W * 2, part_H))
        self.main_uterus_right_icon_click_large_1080p = cv2.resize(main_uterus_right_icon_click_large, (part_W * 2, part_H))
        main_camera_left_icon_click_large = cv2.imread("./sources/gui_fig_v3/main_v2/camera_large_left.png", cv2.IMREAD_UNCHANGED)
        main_camera_right_icon_click_large = cv2.imread("./sources/gui_fig_v3/main_v2/camera_large_right.png", cv2.IMREAD_UNCHANGED)
        self.main_camera_left_icon_click_large_1080p = cv2.resize(main_camera_left_icon_click_large, (part_W * 2, part_H))
        self.main_camera_right_icon_click_large_1080p = cv2.resize(main_camera_right_icon_click_large, (part_W * 2, part_H))

        # Main panel bbox left
        self.main_menu_bbox_left_small_1080p = [0, 0, part_W_l, part_H]
        self.main_menu_bbox_left_large_1080p = [0, 0, 2 * part_W, part_H]
        self.main_uterus_bbox_left_large_1080p = [0, part_H, 2 * part_W, 2 * part_H]
        self.main_camera_bbox_left_large_1080p = [0, 2 * part_H, 2 * part_W, 3 * part_H]
        # Main panel bbox right
        self.main_menu_bbox_right_small_1080p = [resize_W - part_W_l, 0, resize_W, part_H]
        self.main_menu_bbox_right_large_1080p = [resize_W - 2 * part_W, 0, resize_W, part_H]
        self.main_uterus_bbox_right_large_1080p = [resize_W - 2 * part_W, 1 * part_H, resize_W, 2 * part_H]
        self.main_camera_bbox_right_large_1080p = [resize_W - 2 * part_W, 2 * part_H, resize_W, 3 * part_H]
        # endregion

        # region  ------ camera interface buttons ------
        camera_panel_W = round(resize_W / 8 * resize_scale)
        camera_panel_H = round(resize_H / 5)

        camera_panel_large_left_bg = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_click_bg_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_large_right_bg = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_click_bg_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_small_left_bg = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_bg_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_small_right_bg = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_bg_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_small_click_left_bg = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_click_bg_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_small_click_right_bg = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_click_bg_right.png', cv2.IMREAD_UNCHANGED)
        self.camera_panel_large_left_bg_1080p = cv2.resize(camera_panel_large_left_bg, (camera_panel_W, camera_panel_H))
        self.camera_panel_large_right_bg_1080p = cv2.resize(camera_panel_large_right_bg, (camera_panel_W, camera_panel_H))
        self.camera_panel_small_left_bg_1080p = cv2.resize(camera_panel_small_left_bg, (camera_panel_W, camera_panel_H))
        self.camera_panel_small_right_bg_1080p = cv2.resize(camera_panel_small_right_bg, (camera_panel_W, camera_panel_H))
        self.camera_panel_small_click_left_bg_1080p = cv2.resize(camera_panel_small_click_left_bg, (camera_panel_W, camera_panel_H))
        self.camera_panel_small_click_right_bg_1080p = cv2.resize(camera_panel_small_click_right_bg, (camera_panel_W, camera_panel_H))

        # ------------ camera left -----------
        camera_panel_click_icon_large_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_large_left.png', cv2.IMREAD_UNCHANGED)
        self.camera_panel_click_icon_large_left_1080p = cv2.resize(camera_panel_click_icon_large_left, (camera_panel_W, camera_panel_H))

        camera_panel_zoomin_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_in_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomout_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_out_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomauto_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_tool_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_setzero_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_global_left.png', cv2.IMREAD_UNCHANGED)
        self.camera_panel_zoomin_icon_left_1080p = cv2.resize(camera_panel_zoomin_icon_left, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomout_icon_left_1080p = cv2.resize(camera_panel_zoomout_icon_left, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomauto_icon_left_1080p = cv2.resize(camera_panel_zoomauto_icon_left, (camera_panel_W, camera_panel_H))
        self.camera_panel_setzero_icon_left_1080p = cv2.resize(camera_panel_setzero_icon_left, (camera_panel_W, camera_panel_H))

        camera_panel_zoomin_click_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_in_click_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomout_click_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_out_click_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomauto_click_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_tool_click_left.png', cv2.IMREAD_UNCHANGED)
        camera_panel_setzero_click_icon_left = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_global_click_left.png', cv2.IMREAD_UNCHANGED)
        self.camera_panel_zoomin_click_icon_left_1080p = cv2.resize(camera_panel_zoomin_click_icon_left, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomout_click_icon_left_1080p = cv2.resize(camera_panel_zoomout_click_icon_left, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomauto_click_icon_left_1080p = cv2.resize(camera_panel_zoomauto_click_icon_left, (camera_panel_W, camera_panel_H))
        self.camera_panel_setzero_click_icon_left_1080p = cv2.resize(camera_panel_setzero_click_icon_left, (camera_panel_W, camera_panel_H))

        # ------------ camera right -----------
        camera_panel_click_icon_large_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_large_right.png', cv2.IMREAD_UNCHANGED)
        self.camera_panel_click_icon_large_right_1080p = cv2.resize(camera_panel_click_icon_large_right, (camera_panel_W, camera_panel_H))

        camera_panel_zoomin_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_in_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomout_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_out_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomauto_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_tool_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_setzero_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_global_right.png', cv2.IMREAD_UNCHANGED)
        self.camera_panel_zoomin_icon_right_1080p = cv2.resize(camera_panel_zoomin_icon_right, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomout_icon_right_1080p = cv2.resize(camera_panel_zoomout_icon_right, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomauto_icon_right_1080p = cv2.resize(camera_panel_zoomauto_icon_right, (camera_panel_W, camera_panel_H))
        self.camera_panel_setzero_icon_right_1080p = cv2.resize(camera_panel_setzero_icon_right, (camera_panel_W, camera_panel_H))

        camera_panel_zoomin_click_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_in_click_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomout_click_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_out_click_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_zoomauto_click_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_tool_click_right.png', cv2.IMREAD_UNCHANGED)
        camera_panel_setzero_click_icon_right = cv2.imread('./sources/gui_fig_v3/camera_v3/camera_global_click_right.png', cv2.IMREAD_UNCHANGED)
        self.camera_panel_zoomin_click_icon_right_1080p = cv2.resize(camera_panel_zoomin_click_icon_right, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomout_click_icon_right_1080p = cv2.resize(camera_panel_zoomout_click_icon_right, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomauto_click_icon_right_1080p = cv2.resize(camera_panel_zoomauto_click_icon_right, (camera_panel_W, camera_panel_H))
        self.camera_panel_setzero_click_icon_right_1080p = cv2.resize(camera_panel_setzero_click_icon_right, (camera_panel_W, camera_panel_H))

        self.main_camera_panel_bbox_left_large_1080p = [0, 2 * camera_panel_H, camera_panel_W, 3 * camera_panel_H]
        self.main_camera_panel_bbox_right_large_1080p = [resize_W - camera_panel_W, 2 * camera_panel_H, resize_W, 3 * camera_panel_H]

        self.camera_panel_zoomin_bbox_left_1080p = [0, 0, camera_panel_W, camera_panel_H]
        self.camera_panel_zoomout_bbox_left_1080p = [0, camera_panel_H, camera_panel_W, 2 * camera_panel_H]
        self.camera_panel_zoomauto_bbox_left_1080p = [0, 3 * camera_panel_H, camera_panel_W, 4 * camera_panel_H]
        self.camera_panel_setzero_bbox_left_1080p = [0, 4 * camera_panel_H, camera_panel_W, 5 * camera_panel_H]
        self.camera_panel_zoomin_bbox_right_1080p = [resize_W - camera_panel_W, 0, resize_W, camera_panel_H]
        self.camera_panel_zoomout_bbox_right_1080p = [resize_W - camera_panel_W, camera_panel_H, resize_W, 2 * camera_panel_H]
        self.camera_panel_zoomauto_bbox_right_1080p = [resize_W - camera_panel_W, 3 * camera_panel_H, resize_W, 4 * camera_panel_H]
        self.camera_panel_setzero_bbox_right_1080p = [resize_W - camera_panel_W, 4 * camera_panel_H, resize_W,  5 * camera_panel_H]

        # ----- main tool ----------------
        resize_W_tool = round(0.6 * camera_panel_W)
        resize_H_tool = round(0.4 * camera_panel_H)
        main_domain_tool1_icon = cv2.imread("./sources/gui_fig_v3/main/tool1m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool2_icon = cv2.imread("./sources/gui_fig_v3/main/tool2m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool3_icon = cv2.imread("./sources/gui_fig_v3/main/tool3m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool4_icon = cv2.imread("./sources/gui_fig_v3/main/tool4m.png", cv2.IMREAD_UNCHANGED)
        main_domain_tool5_icon = cv2.imread("./sources/gui_fig_v3/main/tool5m.png", cv2.IMREAD_UNCHANGED)
        self.main_domain_tool1_icon_1080p = cv2.resize(main_domain_tool1_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool2_icon_1080p = cv2.resize(main_domain_tool2_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool3_icon_1080p = cv2.resize(main_domain_tool3_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool4_icon_1080p = cv2.resize(main_domain_tool4_icon, (resize_W_tool, resize_H_tool))
        self.main_domain_tool5_icon_1080p = cv2.resize(main_domain_tool5_icon, (resize_W_tool, resize_H_tool))

        # main_tool in terms of icon
        self.camera_panel_tool_icon_bbox_left_large_1080p = [round(0.3 * camera_panel_W), round(0.3 * camera_panel_H),
                                                        round(0.3 * camera_panel_W) + resize_W_tool,
                                                        round(0.3 * camera_panel_H) + resize_H_tool]  #

        self.camera_panel_tool_icon_bbox_right_large_1080p = [camera_panel_W - round(0.3 * camera_panel_W) - resize_W_tool,
                                                              round(0.3 * camera_panel_H),
                                                              camera_panel_W - round(0.3 * camera_panel_W),
                                                              round(0.3 * camera_panel_H) + resize_H_tool]  #
        # endregion

        # region  ------ uterus interface buttons ------
        if self.uterus_interface_mode == 'high':
            uterus_source_dir = 'uterus_v2_high'
            uterus_panel_W = round(resize_W / 6.5 * resize_scale)
            uterus_panel_H = round(resize_H / 20)

            uterus_panel_large_left_bg = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/3_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_large_right_bg = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/3_right.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_large_left_bg_1080p = cv2.resize(uterus_panel_large_left_bg, (uterus_panel_W, uterus_panel_H * 2))
            self.uterus_panel_large_right_bg_1080p = cv2.resize(uterus_panel_large_right_bg, (uterus_panel_W, uterus_panel_H * 2))
            uterus_panel_left_bg = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/4_uterus_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_bg = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/4_uterus_right.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_left_bg_1080p = cv2.resize(uterus_panel_left_bg, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_bg_1080p = cv2.resize(uterus_panel_right_bg, (uterus_panel_W, uterus_panel_H*3))

            uterus_panel_click_icon_large_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/uterus_3_left.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_click_icon_large_left_1080p = cv2.resize(uterus_panel_click_icon_large_left, (uterus_panel_W, uterus_panel_H * 2))
            uterus_panel_click_icon_large_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/uterus_3_right.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_click_icon_large_right_1080p = cv2.resize(uterus_panel_click_icon_large_right, (uterus_panel_W, uterus_panel_H * 2))

            # ------------ uterus left -----------
            uterus_panel_up_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_left.png',  cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_icon_left_1080p = cv2.resize(uterus_panel_up_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_icon_left_1080p = cv2.resize(uterus_panel_down_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_icon_left_1080p = cv2.resize(uterus_panel_left_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_icon_left_1080p = cv2.resize(uterus_panel_right_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_icon_left_1080p = cv2.resize(uterus_panel_insert_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_icon_left_1080p = cv2.resize(uterus_panel_retract_icon_left, (uterus_panel_W, uterus_panel_H*3))

            uterus_panel_up_click_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_click_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_click_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_click_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_click_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_click_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_click_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_click_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_click_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_click_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_click_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_click_left.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_click_icon_left_1080p = cv2.resize(uterus_panel_up_click_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_click_icon_left_1080p = cv2.resize(uterus_panel_down_click_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_click_icon_left_1080p = cv2.resize(uterus_panel_left_click_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_click_icon_left_1080p = cv2.resize(uterus_panel_right_click_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_click_icon_left_1080p = cv2.resize(uterus_panel_insert_click_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_click_icon_left_1080p = cv2.resize(uterus_panel_retract_click_icon_left, (uterus_panel_W, uterus_panel_H*3))

            uterus_panel_up_click_nearly_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_click_nearly_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_click_nearly_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_click_nearly_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_click_nearly_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_click_nearly_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_click_nearly_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_click_nearly_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_click_nearly_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_click_nearly_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_click_nearly_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_click_nearly_left.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_click_nearly_icon_left_1080p = cv2.resize(uterus_panel_up_click_nearly_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_click_nearly_icon_left_1080p = cv2.resize(uterus_panel_down_click_nearly_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_click_nearly_icon_left_1080p = cv2.resize(uterus_panel_left_click_nearly_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_click_nearly_icon_left_1080p = cv2.resize(uterus_panel_right_click_nearly_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_click_nearly_icon_left_1080p = cv2.resize(uterus_panel_insert_click_nearly_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_click_nearly_icon_left_1080p = cv2.resize(uterus_panel_retract_click_nearly_icon_left, (uterus_panel_W, uterus_panel_H*3))

            uterus_panel_up_click_limit_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_click_limit_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_click_limit_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_click_limit_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_click_limit_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_click_limit_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_click_limit_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_click_limit_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_click_limit_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_click_limit_left.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_click_limit_icon_left = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_click_limit_left.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_click_limit_icon_left_1080p = cv2.resize(uterus_panel_up_click_limit_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_click_limit_icon_left_1080p = cv2.resize(uterus_panel_down_click_limit_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_click_limit_icon_left_1080p = cv2.resize(uterus_panel_left_click_limit_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_click_limit_icon_left_1080p = cv2.resize(uterus_panel_right_click_limit_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_click_limit_icon_left_1080p = cv2.resize(uterus_panel_insert_click_limit_icon_left, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_click_limit_icon_left_1080p = cv2.resize(uterus_panel_retract_click_limit_icon_left, (uterus_panel_W, uterus_panel_H*3))

            # ------------ uterus right -----------
            uterus_panel_up_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_right.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_icon_right_1080p = cv2.resize(uterus_panel_up_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_icon_right_1080p = cv2.resize(uterus_panel_down_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_icon_right_1080p = cv2.resize(uterus_panel_left_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_icon_right_1080p = cv2.resize(uterus_panel_right_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_icon_right_1080p = cv2.resize(uterus_panel_insert_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_icon_right_1080p = cv2.resize(uterus_panel_retract_icon_right, (uterus_panel_W, uterus_panel_H*3))

            uterus_panel_up_click_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_click_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_click_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_click_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_click_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_click_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_click_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_click_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_click_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_click_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_click_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_click_right.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_click_icon_right_1080p = cv2.resize(uterus_panel_up_click_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_click_icon_right_1080p = cv2.resize(uterus_panel_down_click_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_click_icon_right_1080p = cv2.resize(uterus_panel_left_click_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_click_icon_right_1080p = cv2.resize(uterus_panel_right_click_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_click_icon_right_1080p = cv2.resize(uterus_panel_insert_click_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_click_icon_right_1080p = cv2.resize(uterus_panel_retract_click_icon_right, (uterus_panel_W, uterus_panel_H*3))

            uterus_panel_up_click_nearly_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_click_nearly_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_click_nearly_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_click_nearly_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_click_nearly_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_click_nearly_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_click_nearly_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_click_nearly_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_click_nearly_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_click_nearly_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_click_nearly_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_click_nearly_right.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_click_nearly_icon_right_1080p = cv2.resize(uterus_panel_up_click_nearly_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_click_nearly_icon_right_1080p = cv2.resize(uterus_panel_down_click_nearly_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_click_nearly_icon_right_1080p = cv2.resize(uterus_panel_left_click_nearly_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_click_nearly_icon_right_1080p = cv2.resize(uterus_panel_right_click_nearly_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_click_nearly_icon_right_1080p = cv2.resize(uterus_panel_insert_click_nearly_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_click_nearly_icon_right_1080p = cv2.resize(uterus_panel_retract_click_nearly_icon_right, (uterus_panel_W, uterus_panel_H*3))

            uterus_panel_up_click_limit_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/up_click_limit_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_down_click_limit_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/down_click_limit_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_left_click_limit_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/left_click_limit_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_right_click_limit_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/right_click_limit_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_insert_click_limit_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/insert_click_limit_right.png', cv2.IMREAD_UNCHANGED)
            uterus_panel_retract_click_limit_icon_right = cv2.imread('./sources/gui_fig_v3/' + uterus_source_dir + '/retract_click_limit_right.png', cv2.IMREAD_UNCHANGED)
            self.uterus_panel_up_click_limit_icon_right_1080p = cv2.resize(uterus_panel_up_click_limit_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_down_click_limit_icon_right_1080p = cv2.resize(uterus_panel_down_click_limit_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_left_click_limit_icon_right_1080p = cv2.resize(uterus_panel_left_click_limit_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_right_click_limit_icon_right_1080p = cv2.resize(uterus_panel_right_click_limit_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_insert_click_limit_icon_right_1080p = cv2.resize(uterus_panel_insert_click_limit_icon_right, (uterus_panel_W, uterus_panel_H*3))
            self.uterus_panel_retract_click_limit_icon_right_1080p = cv2.resize(uterus_panel_retract_click_limit_icon_right, (uterus_panel_W, uterus_panel_H*3))

            self.main_uterus_panel_bbox_left_large_1080p = [0, 0, uterus_panel_W, uterus_panel_H*2]
            self.main_uterus_panel_bbox_right_large_1080p = [resize_W - uterus_panel_W, 0, resize_W, uterus_panel_H*2]

            self.uterus_panel_up_bbox_left_1080p = [0, 2 * uterus_panel_H, uterus_panel_W, 5 * uterus_panel_H]
            self.uterus_panel_down_bbox_left_1080p = [0, 5 * uterus_panel_H, uterus_panel_W, 8 * uterus_panel_H]
            self.uterus_panel_left_bbox_left_1080p = [0, 8 * uterus_panel_H, uterus_panel_W, 11 * uterus_panel_H]
            self.uterus_panel_right_bbox_left_1080p = [0, 11 * uterus_panel_H, uterus_panel_W, 14 * uterus_panel_H]
            self.uterus_panel_insert_bbox_left_1080p = [0, 14 * uterus_panel_H, uterus_panel_W, 17 * uterus_panel_H]
            self.uterus_panel_retract_bbox_left_1080p = [0, 17 * uterus_panel_H, uterus_panel_W, 20 * uterus_panel_H]

            self.uterus_panel_up_bbox_right_1080p = [resize_W - uterus_panel_W, 2 * uterus_panel_H, resize_W, 5 * uterus_panel_H]
            self.uterus_panel_down_bbox_right_1080p = [resize_W - uterus_panel_W, 5 * uterus_panel_H, resize_W, 8 * uterus_panel_H]
            self.uterus_panel_left_bbox_right_1080p = [resize_W - uterus_panel_W, 8 * uterus_panel_H, resize_W,  11 * uterus_panel_H]
            self.uterus_panel_right_bbox_right_1080p = [resize_W - uterus_panel_W, 11 * uterus_panel_H, resize_W, 14 * uterus_panel_H]
            self.uterus_panel_insert_bbox_right_1080p = [resize_W - uterus_panel_W, 14 * uterus_panel_H, resize_W, 17 * uterus_panel_H]
            self.uterus_panel_retract_bbox_right_1080p = [resize_W - uterus_panel_W, 17 * uterus_panel_H, resize_W, 20 * uterus_panel_H]
        # endregion

        # region  ------ setting interface buttons ------
        print('load_uterus_icon_success')

        # ------------ test_setting -----------
        setting_panel_W = round(resize_W / 8 * resize_scale)
        setting_panel_H = round(resize_H / 8)

        setting_panel_left_bg = cv2.imread('./sources/gui_fig_v3/setting/1_setting_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_right_bg = cv2.imread('./sources/gui_fig_v3/setting/1_setting_right.png', cv2.IMREAD_UNCHANGED)
        self.setting_panel_left_bg_1080p = cv2.resize(setting_panel_left_bg, (setting_panel_W, setting_panel_H))
        self.setting_panel_right_bg_1080p = cv2.resize(setting_panel_right_bg, (setting_panel_W, setting_panel_H))

        # ------------ setting left -----------
        setting_panel_lap_0_icon_left = cv2.imread('./sources/gui_fig_v3/setting/laparoscope_0_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_lap_30_icon_left = cv2.imread('./sources/gui_fig_v3/setting/laparoscope_30_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_left_icon_left = cv2.imread('./sources/gui_fig_v3/setting/hand_left_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_right_icon_left = cv2.imread('./sources/gui_fig_v3/setting/hand_right_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_uspeed_icon_left = cv2.imread('./sources/gui_fig_v3/setting/uspeed_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_cspeed_icon_left = cv2.imread('./sources/gui_fig_v3/setting/cspeed_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_box_icon_left = cv2.imread('./sources/gui_fig_v3/setting/box_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_depth_icon_left = cv2.imread('./sources/gui_fig_v3/setting/depth_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_rotation_icon_left = cv2.imread('./sources/gui_fig_v3/setting/rotation_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_tip_usage_icon_left = cv2.imread('./sources/gui_fig_v3/setting/tip_usage_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_global_init_icon_left = cv2.imread('./sources/gui_fig_v3/setting/global_init_left.png', cv2.IMREAD_UNCHANGED)
        self.setting_panel_lap_0_icon_left_1080p = cv2.resize(setting_panel_lap_0_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_lap_30_icon_left_1080p = cv2.resize(setting_panel_lap_30_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_left_icon_left_1080p = cv2.resize(setting_panel_hand_left_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_right_icon_left_1080p = cv2.resize(setting_panel_hand_right_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_uspeed_icon_left_1080p = cv2.resize(setting_panel_uspeed_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_cspeed_icon_left_1080p = cv2.resize(setting_panel_cspeed_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_box_icon_left_1080p = cv2.resize(setting_panel_box_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_depth_icon_left_1080p = cv2.resize(setting_panel_depth_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_rotation_icon_left_1080p = cv2.resize(setting_panel_rotation_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_tip_usage_icon_left_1080p = cv2.resize(setting_panel_tip_usage_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_global_init_icon_left_1080p = cv2.resize(setting_panel_global_init_icon_left, (setting_panel_W, setting_panel_H))

        setting_panel_lap_0_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/laparoscope_0_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_lap_30_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/laparoscope_30_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_left_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/hand_left_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_right_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/hand_right_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_uspeed_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/uspeed_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_cspeed_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/cspeed_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_box_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/box_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_depth_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/depth_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_rotation_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/rotation_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_tip_usage_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/tip_usage_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_panel_global_init_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting/global_init_click_left.png', cv2.IMREAD_UNCHANGED)
        self.setting_panel_lap_0_click_icon_left_1080p = cv2.resize(setting_panel_lap_0_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_lap_30_click_icon_left_1080p = cv2.resize(setting_panel_lap_30_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_left_click_icon_left_1080p = cv2.resize(setting_panel_hand_left_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_right_click_icon_left_1080p = cv2.resize(setting_panel_hand_right_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_uspeed_click_icon_left_1080p = cv2.resize(setting_panel_uspeed_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_cspeed_click_icon_left_1080p = cv2.resize(setting_panel_cspeed_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_box_click_icon_left_1080p = cv2.resize(setting_panel_box_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_depth_click_icon_left_1080p = cv2.resize(setting_panel_depth_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_rotation_click_icon_left_1080p = cv2.resize(setting_panel_rotation_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_tip_usage_click_icon_left_1080p = cv2.resize(setting_panel_tip_usage_click_icon_left, (setting_panel_W, setting_panel_H))
        self.setting_panel_global_init_click_icon_left_1080p = cv2.resize(setting_panel_global_init_click_icon_left, (setting_panel_W, setting_panel_H))

        # # ------------ setting right -----------
        setting_panel_lap_0_icon_right = cv2.imread('./sources/gui_fig_v3/setting/laparoscope_0_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_lap_30_icon_right = cv2.imread('./sources/gui_fig_v3/setting/laparoscope_30_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_left_icon_right = cv2.imread('./sources/gui_fig_v3/setting/hand_left_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_right_icon_right = cv2.imread('./sources/gui_fig_v3/setting/hand_right_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_uspeed_icon_right = cv2.imread('./sources/gui_fig_v3/setting/uspeed_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_cspeed_icon_right = cv2.imread('./sources/gui_fig_v3/setting/cspeed_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_box_icon_right = cv2.imread('./sources/gui_fig_v3/setting/box_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_depth_icon_right = cv2.imread('./sources/gui_fig_v3/setting/depth_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_rotation_icon_right = cv2.imread('./sources/gui_fig_v3/setting/rotation_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_tip_usage_icon_right = cv2.imread('./sources/gui_fig_v3/setting/tip_usage_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_global_init_icon_right = cv2.imread('./sources/gui_fig_v3/setting/global_init_right.png', cv2.IMREAD_UNCHANGED)
        self.setting_panel_lap_0_icon_right_1080p = cv2.resize(setting_panel_lap_0_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_lap_30_icon_right_1080p = cv2.resize(setting_panel_lap_30_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_left_icon_right_1080p = cv2.resize(setting_panel_hand_left_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_right_icon_right_1080p = cv2.resize(setting_panel_hand_right_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_uspeed_icon_right_1080p = cv2.resize(setting_panel_uspeed_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_cspeed_icon_right_1080p = cv2.resize(setting_panel_cspeed_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_box_icon_right_1080p = cv2.resize(setting_panel_box_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_depth_icon_right_1080p = cv2.resize(setting_panel_depth_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_rotation_icon_right_1080p = cv2.resize(setting_panel_rotation_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_tip_usage_icon_right_1080p = cv2.resize(setting_panel_tip_usage_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_global_init_icon_right_1080p = cv2.resize(setting_panel_global_init_icon_right, (setting_panel_W, setting_panel_H))

        setting_panel_lap_0_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/laparoscope_0_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_lap_30_click_icon_right = cv2.imread( './sources/gui_fig_v3/setting/laparoscope_30_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_left_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/hand_left_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_hand_right_click_icon_right = cv2.imread( './sources/gui_fig_v3/setting/hand_right_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_uspeed_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/uspeed_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_cspeed_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/cspeed_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_box_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/box_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_depth_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/depth_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_rotation_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/rotation_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_tip_usage_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/tip_usage_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_panel_global_init_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting/global_init_click_right.png', cv2.IMREAD_UNCHANGED)
        self.setting_panel_lap_0_click_icon_right_1080p = cv2.resize(setting_panel_lap_0_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_lap_30_click_icon_right_1080p = cv2.resize(setting_panel_lap_30_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_left_click_icon_right_1080p = cv2.resize(setting_panel_hand_left_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_hand_right_click_icon_right_1080p = cv2.resize(setting_panel_hand_right_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_uspeed_click_icon_right_1080p = cv2.resize(setting_panel_uspeed_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_cspeed_click_icon_right_1080p = cv2.resize(setting_panel_cspeed_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_box_click_icon_right_1080p = cv2.resize(setting_panel_box_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_depth_click_icon_right_1080p = cv2.resize(setting_panel_depth_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_rotation_click_icon_right_1080p = cv2.resize(setting_panel_rotation_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_tip_usage_click_icon_right_1080p = cv2.resize(setting_panel_tip_usage_click_icon_right, (setting_panel_W, setting_panel_H))
        self.setting_panel_global_init_click_icon_right_1080p = cv2.resize(setting_panel_global_init_click_icon_right, (setting_panel_W, setting_panel_H))

        self.setting_panel_lap_bbox_left_1080p = [0, 0, setting_panel_W, setting_panel_H]
        self.setting_panel_hand_bbox_left_1080p = [0, setting_panel_H, setting_panel_W, 2 * setting_panel_H]
        self.setting_panel_uspeed_bbox_left_1080p = [0, 2 * setting_panel_H, setting_panel_W, 3 * setting_panel_H]
        self.setting_panel_cspeed_bbox_left_1080p = [0, 3 * setting_panel_H, setting_panel_W, 4 * setting_panel_H]
        self.setting_panel_box_bbox_left_1080p = [0, 4 * setting_panel_H, setting_panel_W, 5 * setting_panel_H]
        self.setting_panel_depth_bbox_left_1080p = [0, 5 * setting_panel_H, setting_panel_W, 6 * setting_panel_H]
        self.setting_panel_rotation_bbox_left_1080p = [0, 6 * setting_panel_H, setting_panel_W, 7 * setting_panel_H]
        self.setting_panel_tip_usage_bbox_left_1080p = [0, 7 * setting_panel_H, setting_panel_W, 8 * setting_panel_H]

        self.setting_panel_lap_bbox_right_1080p = [resize_W - setting_panel_W, 0, resize_W, setting_panel_H]
        self.setting_panel_hand_bbox_right_1080p = [resize_W - setting_panel_W, setting_panel_H, resize_W, 2 * setting_panel_H]
        self.setting_panel_uspeed_bbox_right_1080p = [resize_W - setting_panel_W, 2 * setting_panel_H, resize_W, 3 * setting_panel_H]
        self.setting_panel_cspeed_bbox_right_1080p = [resize_W - setting_panel_W, 3 * setting_panel_H, resize_W, 4 * setting_panel_H]
        self.setting_panel_box_bbox_right_1080p = [resize_W - setting_panel_W, 4 * setting_panel_H, resize_W, 5 * setting_panel_H]
        self.setting_panel_depth_bbox_right_1080p = [resize_W - setting_panel_W, 5 * setting_panel_H, resize_W,  6 * setting_panel_H]
        self.setting_panel_rotation_bbox_right_1080p = [resize_W - setting_panel_W, 6 * setting_panel_H, resize_W, 7 * setting_panel_H]
        self.setting_panel_tip_usage_bbox_right_1080p = [resize_W - setting_panel_W, 7 * setting_panel_H, resize_W, 8 * setting_panel_H]

        self.setting_font_1080p = resize_W * 4.5e-4
        self.setting_font_large_1080p = resize_W * 5e-4
        # endregion

        # region ------ setting uterus ------
        setting_uterus_panel_W = round(resize_W / 8)
        setting_uterus_panel_H = round(resize_H / 8)

        setting_uterus_panel_left_bg= cv2.imread('./sources/gui_fig_v3/setting_uterus/1_setting_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_right_bg = cv2.imread('./sources/gui_fig_v3/setting_uterus/1_setting_right.png', cv2.IMREAD_UNCHANGED)
        self.setting_uterus_panel_left_bg_1080p = cv2.resize(setting_uterus_panel_left_bg, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_right_bg_1080p = cv2.resize(setting_uterus_panel_right_bg, (setting_uterus_panel_W, setting_uterus_panel_H))

        # ------------ setting uterus left -----------
        setting_uterus_panel_lap_0_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_0_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lap_30_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_30_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_left_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_left_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_right_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_right_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_uspeed_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/uspeed_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_tip_usage_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/tip_usage_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_anteversion_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/anteversion_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lateral_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/lateral_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_insertion_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/insertion_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_global_init_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/global_init_left.png', cv2.IMREAD_UNCHANGED)
        self.setting_uterus_panel_lap_0_icon_left_1080p = cv2.resize(setting_uterus_panel_lap_0_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lap_30_icon_left_1080p = cv2.resize(setting_uterus_panel_lap_30_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_left_icon_left_1080p = cv2.resize(setting_uterus_panel_hand_left_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_right_icon_left_1080p = cv2.resize(setting_uterus_panel_hand_right_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_uspeed_icon_left_1080p = cv2.resize(setting_uterus_panel_uspeed_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_tip_usage_icon_left_1080p = cv2.resize(setting_uterus_panel_tip_usage_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_anteversion_icon_left_1080p = cv2.resize(setting_uterus_panel_anteversion_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lateral_icon_left_1080p = cv2.resize(setting_uterus_panel_lateral_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_insertion_icon_left_1080p = cv2.resize(setting_uterus_panel_insertion_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_insertion_icon_left_1080p = cv2.resize(setting_uterus_panel_insertion_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_global_init_icon_left_1080p = cv2.resize(setting_uterus_panel_global_init_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))

        setting_uterus_panel_lap_0_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_0_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lap_30_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_30_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_left_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_left_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_right_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_right_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_uspeed_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/uspeed_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_tip_usage_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/tip_usage_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_anteversion_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/anteversion_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lateral_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/lateral_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_insertion_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/insertion_click_left.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_global_init_click_icon_left = cv2.imread('./sources/gui_fig_v3/setting_uterus/global_init_click_left.png', cv2.IMREAD_UNCHANGED)
        self.setting_uterus_panel_lap_0_click_icon_left_1080p = cv2.resize(setting_uterus_panel_lap_0_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lap_30_click_icon_left_1080p = cv2.resize(setting_uterus_panel_lap_30_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_left_click_icon_left_1080p = cv2.resize(setting_uterus_panel_hand_left_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_right_click_icon_left_1080p = cv2.resize(setting_uterus_panel_hand_right_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_uspeed_click_icon_left_1080p = cv2.resize(setting_uterus_panel_uspeed_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_tip_usage_click_icon_left_1080p = cv2.resize(setting_uterus_panel_tip_usage_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_anteversion_click_icon_left_1080p = cv2.resize(setting_uterus_panel_anteversion_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lateral_click_icon_left_1080p = cv2.resize(setting_uterus_panel_lateral_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_insertion_click_icon_left_1080p = cv2.resize(setting_uterus_panel_insertion_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_global_init_click_icon_left_1080p = cv2.resize(setting_uterus_panel_global_init_click_icon_left, (setting_uterus_panel_W, setting_uterus_panel_H))

        # ------------ setting uterus right -----------
        setting_uterus_panel_lap_0_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_0_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lap_30_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_30_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_left_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_left_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_right_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_right_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_uspeed_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/uspeed_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_tip_usage_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/tip_usage_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_anteversion_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/anteversion_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lateral_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/lateral_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_insertion_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/insertion_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_global_init_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/global_init_right.png', cv2.IMREAD_UNCHANGED)
        self.setting_uterus_panel_lap_0_icon_right_1080p = cv2.resize(setting_uterus_panel_lap_0_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lap_30_icon_right_1080p = cv2.resize(setting_uterus_panel_lap_30_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_left_icon_right_1080p = cv2.resize(setting_uterus_panel_hand_left_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_right_icon_right_1080p = cv2.resize(setting_uterus_panel_hand_right_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_uspeed_icon_right_1080p = cv2.resize(setting_uterus_panel_uspeed_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_tip_usage_icon_right_1080p = cv2.resize(setting_uterus_panel_tip_usage_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_anteversion_icon_right_1080p = cv2.resize(setting_uterus_panel_anteversion_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lateral_icon_right_1080p = cv2.resize(setting_uterus_panel_lateral_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_insertion_icon_right_1080p = cv2.resize(setting_uterus_panel_insertion_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_global_init_icon_right_1080p = cv2.resize(setting_uterus_panel_global_init_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))

        setting_uterus_panel_lap_0_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_0_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lap_30_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/laparoscope_30_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_left_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_left_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_hand_right_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/hand_right_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_uspeed_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/uspeed_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_tip_usage_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/tip_usage_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_anteversion_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/anteversion_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_lateral_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/lateral_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_insertion_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/insertion_click_right.png', cv2.IMREAD_UNCHANGED)
        setting_uterus_panel_global_init_click_icon_right = cv2.imread('./sources/gui_fig_v3/setting_uterus/global_init_click_right.png', cv2.IMREAD_UNCHANGED)
        self.setting_uterus_panel_lap_0_click_icon_right_1080p = cv2.resize(setting_uterus_panel_lap_0_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lap_30_click_icon_right_1080p = cv2.resize(setting_uterus_panel_lap_30_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_left_click_icon_right_1080p = cv2.resize(setting_uterus_panel_hand_left_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_hand_right_click_icon_right_1080p = cv2.resize(setting_uterus_panel_hand_right_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_uspeed_click_icon_right_1080p = cv2.resize(setting_uterus_panel_uspeed_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_tip_usage_click_icon_right_1080p = cv2.resize(setting_uterus_panel_tip_usage_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_anteversion_click_icon_right_1080p = cv2.resize(setting_uterus_panel_anteversion_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_lateral_click_icon_right_1080p = cv2.resize(setting_uterus_panel_lateral_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_insertion_click_icon_right_1080p = cv2.resize(setting_uterus_panel_insertion_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))
        self.setting_uterus_panel_global_init_click_icon_right_1080p = cv2.resize(setting_uterus_panel_global_init_click_icon_right, (setting_uterus_panel_W, setting_uterus_panel_H))

        # ------------ setting uterus bbox -----------
        self.setting_uterus_panel_one_eight_left_1080p = [0, 0, setting_uterus_panel_W, setting_uterus_panel_H]  # x1, y1, x2, y2
        self.setting_uterus_panel_two_eight_left_1080p = [0, setting_uterus_panel_H, setting_uterus_panel_W, 2 * setting_uterus_panel_H]  # x1, y1, x2, y2
        self.setting_uterus_panel_three_eight_left_1080p = [0, 2 * setting_uterus_panel_H, setting_uterus_panel_W, 3 * setting_uterus_panel_H]  # x1, y1, x2, y2
        self.setting_uterus_panel_four_eight_left_1080p = [0, 3 * setting_uterus_panel_H, setting_uterus_panel_W, 4 * setting_uterus_panel_H]  # x1, y1, x2, y2
        self.setting_uterus_panel_five_eight_left_1080p = [0, 4 * setting_uterus_panel_H, setting_uterus_panel_W, 5 * setting_uterus_panel_H]  # x1, y1, x2, y2
        self.setting_uterus_panel_six_eight_left_1080p = [0, 5 * setting_uterus_panel_H, setting_uterus_panel_W, 6 * setting_uterus_panel_H]  # x1, y1, x2, y2
        self.setting_uterus_panel_seven_eight_left_1080p = [0, 6 * setting_uterus_panel_H, setting_uterus_panel_W, 7 * setting_uterus_panel_H]  # x1, y1, x2, y2

        self.setting_uterus_panel_one_eight_right_1080p = [resize_W - setting_uterus_panel_W, 0, resize_W, setting_uterus_panel_H]
        self.setting_uterus_panel_two_eight_right_1080p = [resize_W - setting_uterus_panel_W, setting_uterus_panel_H, resize_W, 2 * setting_uterus_panel_H]
        self.setting_uterus_panel_three_eight_right_1080p = [resize_W - setting_uterus_panel_W, 2 * setting_uterus_panel_H, resize_W, 3 * setting_uterus_panel_H]
        self.setting_uterus_panel_four_eight_right_1080p = [resize_W - setting_uterus_panel_W, 3 * setting_uterus_panel_H, resize_W, 4 * setting_uterus_panel_H]
        self.setting_uterus_panel_five_eight_right_1080p = [resize_W - setting_uterus_panel_W, 4 * setting_uterus_panel_H, resize_W, 5 * setting_uterus_panel_H]
        self.setting_uterus_panel_six_eight_right_1080p = [resize_W - setting_uterus_panel_W, 5 * setting_uterus_panel_H, resize_W, 6 * setting_uterus_panel_H]
        self.setting_uterus_panel_seven_eight_right_1080p = [resize_W - setting_uterus_panel_W, 6 * setting_uterus_panel_H, resize_W, 7 * setting_uterus_panel_H]
        # endregion
        print('load_icons for panel 1080p done ')

    def load_template_1080p(self):
        print('load_1080_template')
        # region ---------- load main template -----------------
        paste_template(self.main_icon_small_bg_1080p, self.main_menu_icon_small_1080p, alpha=self.alpha_main_small_bg)

        paste_template(self.main_icon_large_left_bg_1080p, self.main_menu_left_icon_large_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.main_icon_large_left_bg_1080p, self.main_uterus_left_icon_large_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.main_icon_large_left_bg_1080p, self.main_camera_left_icon_large_1080p, alpha=self.alpha_main_large_bg)

        paste_template(self.main_icon_large_left_bg_1080p, self.main_menu_right_icon_large_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.main_icon_large_left_bg_1080p, self.main_uterus_right_icon_large_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.main_icon_large_left_bg_1080p, self.main_camera_right_icon_large_1080p, alpha=self.alpha_main_large_bg)

        paste_template(self.main_left_icon_click_large_bg_1080p, self.main_uterus_left_icon_click_large_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.main_left_icon_click_large_bg_1080p, self.main_camera_left_icon_click_large_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.main_right_icon_click_large_bg_1080p, self.main_uterus_right_icon_click_large_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.main_right_icon_click_large_bg_1080p, self.main_camera_right_icon_click_large_1080p, alpha=self.alpha_main_large_bg)

        paste_template(self.uterus_panel_large_left_bg_1080p, self.uterus_panel_click_icon_large_left_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.uterus_panel_large_right_bg_1080p, self.uterus_panel_click_icon_large_right_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.camera_panel_large_left_bg_1080p, self.camera_panel_click_icon_large_left_1080p, alpha=self.alpha_main_large_bg)
        paste_template(self.camera_panel_large_right_bg_1080p, self.camera_panel_click_icon_large_right_1080p, alpha=self.alpha_main_large_bg)
        print('load main done')
        # endregion

        # region ---------- load uterus template -----------------
        if self.uterus_interface_mode == 'high' or self.uterus_interface_mode == 'wide':
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_up_icon_left_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_down_icon_left_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_left_icon_left_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_right_icon_left_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_insert_icon_left_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_retract_icon_left_1080p, alpha=self.alpha_each_bg)

            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_up_icon_right_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_down_icon_right_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_left_icon_right_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_right_icon_right_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_insert_icon_right_1080p, alpha=self.alpha_each_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_retract_icon_right_1080p, alpha=self.alpha_each_bg)

            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_up_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_down_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_left_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_right_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_insert_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_retract_click_icon_left_1080p, alpha=self.alpha_each_click_bg)

            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_up_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_down_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_left_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_right_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_insert_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_retract_click_icon_right_1080p, alpha=self.alpha_each_click_bg)

            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_up_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_down_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_left_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_right_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_insert_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_retract_click_nearly_icon_left_1080p, alpha=self.alpha_each_click_bg)

            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_up_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_down_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_left_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_right_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_insert_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_retract_click_nearly_icon_right_1080p, alpha=self.alpha_each_click_bg)

            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_up_click_limit_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_down_click_limit_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_left_click_limit_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_right_click_limit_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_insert_click_limit_icon_left_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_left_bg_1080p, self.uterus_panel_retract_click_limit_icon_left_1080p, alpha=self.alpha_each_click_bg)

            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_up_click_limit_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_down_click_limit_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_left_click_limit_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_right_click_limit_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_insert_click_limit_icon_right_1080p, alpha=self.alpha_each_click_bg)
            paste_template(self.uterus_panel_right_bg_1080p, self.uterus_panel_retract_click_limit_icon_right_1080p, alpha=self.alpha_each_click_bg)

        # endregion

        # region ---------- load camera template -----------------
        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_zoomin_icon_left_1080p, alpha=0.7)
        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_zoomout_icon_left_1080p, alpha=0.7)
        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_zoomauto_icon_left_1080p, alpha=0.7)
        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_setzero_icon_left_1080p, alpha=0.7)

        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_zoomin_click_icon_left_1080p, alpha=0.8)
        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_zoomout_click_icon_left_1080p, alpha=0.8)
        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_zoomauto_click_icon_left_1080p, alpha=0.8)
        paste_template(self.camera_panel_small_left_bg_1080p, self.camera_panel_setzero_click_icon_left_1080p, alpha=0.8)

        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_zoomin_icon_right_1080p, alpha=0.7)
        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_zoomout_icon_right_1080p, alpha=0.7)
        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_zoomauto_icon_right_1080p, alpha=0.7)
        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_setzero_icon_right_1080p, alpha=0.7)

        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_zoomin_click_icon_right_1080p, alpha=0.8)
        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_zoomout_click_icon_right_1080p, alpha=0.8)
        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_zoomauto_click_icon_right_1080p, alpha=0.8)
        paste_template(self.camera_panel_small_right_bg_1080p, self.camera_panel_setzero_click_icon_right_1080p, alpha=0.8)
        print('load main done')

        # ---------- load main tool template  todo
        self.main_domain_tool1_icon_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool1_icon_1080p, alpha=1.0)
        self.main_domain_tool2_icon_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool2_icon_1080p, alpha=1.0)
        self.main_domain_tool3_icon_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool3_icon_1080p, alpha=1.0)
        self.main_domain_tool4_icon_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool4_icon_1080p, alpha=1.0)
        self.main_domain_tool5_icon_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool5_icon_1080p, alpha=1.0)

        self.main_domain_tool_ls_1080p_left = [self.main_domain_tool1_icon_1080p_left, self.main_domain_tool2_icon_1080p_left,
                                               self.main_domain_tool3_icon_1080p_left, self.main_domain_tool4_icon_1080p_left,
                                               self.main_domain_tool5_icon_1080p_left]

        self.main_domain_tool1_icon_click_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool1_icon_1080p, alpha=1.0)
        self.main_domain_tool2_icon_click_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool2_icon_1080p, alpha=1.0)
        self.main_domain_tool3_icon_click_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool3_icon_1080p, alpha=1.0)
        self.main_domain_tool4_icon_click_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool4_icon_1080p, alpha=1.0)
        self.main_domain_tool5_icon_click_1080p_left = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_left_1080p, self.camera_panel_tool_icon_bbox_left_large_1080p,
                                                           self.main_domain_tool5_icon_1080p, alpha=1.0)

        self.main_domain_tool_click_ls_1080p_left = [self.main_domain_tool1_icon_click_1080p_left, self.main_domain_tool2_icon_click_1080p_left,
                                               self.main_domain_tool3_icon_click_1080p_left, self.main_domain_tool4_icon_click_1080p_left,
                                               self.main_domain_tool5_icon_click_1080p_left]

        self.main_domain_tool1_icon_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool1_icon_1080p, alpha=1.0)
        self.main_domain_tool2_icon_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool2_icon_1080p, alpha=1.0)
        self.main_domain_tool3_icon_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool3_icon_1080p, alpha=1.0)
        self.main_domain_tool4_icon_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool4_icon_1080p, alpha=1.0)
        self.main_domain_tool5_icon_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool5_icon_1080p, alpha=1.0)

        self.main_domain_tool_ls_1080p_right = [self.main_domain_tool1_icon_1080p_right, self.main_domain_tool2_icon_1080p_right,
                                               self.main_domain_tool3_icon_1080p_right, self.main_domain_tool4_icon_1080p_right,
                                               self.main_domain_tool5_icon_1080p_right]

        self.main_domain_tool1_icon_click_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool1_icon_1080p, alpha=1.0)
        self.main_domain_tool2_icon_click_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool2_icon_1080p, alpha=1.0)
        self.main_domain_tool3_icon_click_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool3_icon_1080p, alpha=1.0)
        self.main_domain_tool4_icon_click_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool4_icon_1080p, alpha=1.0)
        self.main_domain_tool5_icon_click_1080p_right = paste_bbox_bg_img(self.camera_panel_zoomauto_click_icon_right_1080p, self.camera_panel_tool_icon_bbox_right_large_1080p,
                                                           self.main_domain_tool5_icon_1080p, alpha=1.0)

        self.main_domain_tool_click_ls_1080p_right = [self.main_domain_tool1_icon_click_1080p_right, self.main_domain_tool2_icon_click_1080p_right,
                                               self.main_domain_tool3_icon_click_1080p_right, self.main_domain_tool4_icon_click_1080p_right,
                                               self.main_domain_tool5_icon_click_1080p_right]
        print('load main tool done')
        # endregion

        # region ---------- load setting template -----------------
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_lap_0_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_lap_30_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_hand_left_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_hand_right_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_uspeed_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_cspeed_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_box_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_depth_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_rotation_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_tip_usage_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_global_init_icon_left_1080p, alpha=self.alpha_each_bg)

        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_lap_0_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_lap_30_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_hand_left_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_hand_right_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_uspeed_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_cspeed_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_box_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_depth_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_rotation_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_tip_usage_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_left_bg_1080p, self.setting_panel_global_init_click_icon_left_1080p, alpha=self.alpha_each_click_bg)

        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_lap_0_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_lap_30_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_hand_left_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_hand_right_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_uspeed_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_cspeed_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_box_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_depth_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_rotation_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_tip_usage_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_global_init_icon_right_1080p, alpha=self.alpha_each_bg)

        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_lap_0_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_lap_30_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_hand_left_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_hand_right_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_uspeed_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_cspeed_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_box_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_depth_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_rotation_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_tip_usage_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_panel_right_bg_1080p, self.setting_panel_global_init_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        # endregion

        # region ------ setting uterus template ------
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_lap_0_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_lap_30_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_hand_left_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_hand_right_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_uspeed_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_tip_usage_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_anteversion_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_lateral_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_insertion_icon_left_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_global_init_icon_left_1080p, alpha=self.alpha_each_bg)

        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_lap_0_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_lap_30_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_hand_left_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_hand_right_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_uspeed_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_tip_usage_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_anteversion_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_lateral_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_insertion_click_icon_left_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_left_bg_1080p, self.setting_uterus_panel_global_init_click_icon_left_1080p, alpha=self.alpha_each_click_bg)

        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_lap_0_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_lap_30_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_hand_left_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_hand_right_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_uspeed_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_tip_usage_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_anteversion_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_lateral_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_insertion_icon_right_1080p, alpha=self.alpha_each_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_global_init_icon_right_1080p, alpha=self.alpha_each_bg)

        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_lap_0_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_lap_30_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_hand_left_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_hand_right_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_uspeed_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_tip_usage_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_anteversion_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_lateral_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_insertion_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        paste_template(self.setting_uterus_panel_right_bg_1080p, self.setting_uterus_panel_global_init_click_icon_right_1080p, alpha=self.alpha_each_click_bg)
        # endregion
        print('load 1080p template done')


def add_alpha_channel(img):
    """ add alpha channel to jpg image """

    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new


def paste_bbox_img(img, iou, template, alpha):
    """ 将png透明图像与jpg图像叠加
        y1,y2,x1,x2为叠加位置坐标值
    """

    img[iou[1]:iou[3], iou[0]:iou[2], ::-1] = cv2.addWeighted(img[iou[1]:iou[3], iou[0]:iou[2], ::-1], 1 - alpha,
                                                          template[:, :, :3], alpha, 0)

    return img

    # """ 将png透明图像与jpg图像叠加
    #     y1,y2,x1,x2为叠加位置坐标值
    #     img1: raw image
    #     iou: the merge area of the raw image
    #     template: the target shown images
    #     alpha: weight of raw images
    # """
    # # add background
    # alpha_png = template[:, :, 3] / 255.0 * alpha
    # alpha_png = np.expand_dims(alpha_png, axis=2)
    # alpha_png = np.repeat(alpha_png, 3, axis=2)
    # alpha_jpg = 1 - alpha_png
    #
    # img_result = img.copy()
    # img_result[iou[1]:iou[3], iou[0]:iou[2], :3] = ((alpha_jpg * img[iou[1]:iou[3], iou[0]:iou[2], :3]) +
    #                                           (alpha_png * template[:, :, 2::-1]))
    #
    # # img_result = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB)
    # return img_result


def paste_template(bg, fg, alpha):
    """ 将两个png透明图像叠加"""
    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = bg[:, :, 3] / 255.0 * alpha

    fg[:, :, 3] = (alpha_fg + alpha_bg * (1 - alpha_fg)) * 255.0

    for c in range(0, 3):
        fg[:, :, c] = (alpha_bg * bg[:, :, c] * (1 - alpha_fg) + fg[:, :, c] * alpha_fg) / fg[:, :, 3] * 255.0
        fg[:, :, c][fg[:, :, 3] == 0] = 0
    #
    # return fg


def paste_bbox_bg_img(img1, iou, template, alpha):
    """ 将png透明图像与jpg图像叠加
        y1,y2,x1,x2为叠加位置坐标值
        img1: raw image
        iou: the merge area of the raw image
        template: the target shown images
        alpha: weight of raw images
    """
    # add background
    alpha_png = template[:, :, 3] / 255.0 * alpha
    alpha_png = np.expand_dims(alpha_png, axis=2)
    alpha_png = np.repeat(alpha_png, 3, axis=2)
    alpha_jpg = 1 - alpha_png

    img_result = img1.copy()
    img_result[iou[1]:iou[3], iou[0]:iou[2], :3] = ((alpha_jpg * img1[iou[1]:iou[3], iou[0]:iou[2], :3]) +
                                              (alpha_png * template[:, :, :3]))

    # img_result = cv2.cvtColor(img1, cv2.COLOR_RGBA2RGB)
    return img_result


def paste_bbox_bg_img_old(img1, iou, bg, template, alpha, beta):
    """ 将png透明图像与jpg图像叠加
        y1,y2,x1,x2为叠加位置坐标值
        img1: raw image
        iou: the merge area of the raw image
        bg: the background color
        template: the target shown images
        alpha: weight of raw images
        beta: weight of template
    """
    # Determine if the jpg image is already 4-channel
    if img1.shape[2] == 3:
        img1 = add_alpha_channel(img1)

    # add background
    alpha_png = bg[0:bg.shape[0], 0:bg.shape[1], 3] / 255.0 * alpha
    alpha_png = np.expand_dims(alpha_png, axis=2)
    alpha_png = np.repeat(alpha_png, 3, axis=2)
    alpha_jpg = 1 - alpha_png

    # for c in range(0, 3):
    #     img1[iou[1]:iou[3], iou[0]:iou[2], c] = ((alpha_jpg * img1[iou[1]:iou[3], iou[0]:iou[2], c]) +
    #                                              (alpha_png * bg[0:bg.shape[0], 0:bg.shape[1], c]))
    # print('***********************************************')
    # print(alpha_jpg)
    # print('debug1: ', (alpha_jpg * img1[iou[1]:iou[3], iou[0]:iou[2], :3]).shape)
    # print('debug2: ', (alpha_png * bg[0:bg.shape[0], 0:bg.shape[1], :3]).shape)
    img1[iou[1]:iou[3], iou[0]:iou[2], :3] = ((alpha_jpg * img1[iou[1]:iou[3], iou[0]:iou[2], :3]) +
                                              (alpha_png * bg[0:bg.shape[0], 0:bg.shape[1], :3]))

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGR)
    if img1.shape[2] == 3:
        img1 = add_alpha_channel(img1)

    # obtain the alpha value of the template image, divide the pixel value by 255 to keep the value between 0 and 1
    beta_png = template[0:template.shape[0], 0:template.shape[1], 3] / 255.0 * beta
    beta_png = np.expand_dims(beta_png, axis=2)
    beta_png = np.repeat(beta_png, 3, axis=2)
    beta_jpg = 1 - beta_png

    # add image
    # for c in range(0, 3):
    #     img1[iou[1]:iou[3], iou[0]:iou[2], c] = ((beta_jpg * img1[iou[1]:iou[3], iou[0]:iou[2], c]) +
    #                                              (beta_png * template[0:template.shape[0], 0:template.shape[1], c]))
    img1[iou[1]:iou[3], iou[0]:iou[2], :3] = ((beta_jpg * img1[iou[1]:iou[3], iou[0]:iou[2], :3]) +
                                              (beta_png * template[0:template.shape[0], 0:template.shape[1], :3]))

    img_result = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGR)
    return img_result


def main_aux_point_icon_plot(img, point, template, img_size):
    img_result = img.copy()

    plot_x1 = point[0] - int(template.shape[0]/2)
    plot_y1 = point[1] - int(template.shape[1]/2)
    plot_x2 = point[0] + int(template.shape[0]/2)
    plot_y2 = point[1] + int(template.shape[1]/2)

    if plot_x1 >= 0 and plot_y1 >=0 and plot_x2 <= img_size[0] and plot_y2 <= img_size[1]:
        distance_x = abs(img_size[0]/2 - point[0])/(img_size[0]/2)
        distance_y = abs(img_size[1]/2 - point[1])/(img_size[1]/2)
        distance = max(distance_x, distance_y)

        point_thres_in = 0.5
        point_thres_out = 0.9
        alpha_thres = 0.35

        if distance <= point_thres_in:
            alpha = alpha_thres
        elif distance >= point_thres_out:
            alpha = 1
        else:
            alpha = (distance - point_thres_in)/(point_thres_out-point_thres_in)*(1-alpha_thres) + alpha_thres

        iou = [plot_x1, plot_y1, plot_x2, plot_y2]

        alpha_png = template[:, :, 3] / 255.0 * alpha
        alpha_png = np.expand_dims(alpha_png, axis=2)
        alpha_png = np.repeat(alpha_png, 3, axis=2)
        alpha_jpg = 1 - alpha_png

        img_result[iou[1]:iou[3], iou[0]:iou[2], :3] = ((alpha_jpg * img[iou[1]:iou[3], iou[0]:iou[2], :3]) +
                                                  (alpha_png * template[:, :, 2::-1]))
    return img_result


def main_aux_point_icon_plot_1080p(img, point, template, img_size):
    img_result = img.copy()

    distance_x = abs(img_size[0]/2 - point[0])/(img_size[0]/2)
    distance_y = abs(img_size[1]/2 - point[1])/(img_size[1]/2)
    distance = max(distance_x, distance_y)

    point_thres_in = 0.5
    point_thres_out = 0.9
    alpha_thres = 0

    if distance <= point_thres_in:
        pass
    else:
        plot_x1 = point[0] - int(template.shape[0]/2)
        plot_y1 = point[1] - int(template.shape[1]/2)
        plot_x2 = point[0] + int(template.shape[0]/2)
        plot_y2 = point[1] + int(template.shape[1]/2)

        if plot_x1 >= 0 and plot_y1 >=0 and plot_x2 <= img_size[0] and plot_y2 <= img_size[1]:

            if distance >= point_thres_out:
                alpha = 1
            else:
                alpha = (distance - point_thres_in)/(point_thres_out-point_thres_in)*(1-alpha_thres) + alpha_thres

            iou = [plot_x1, plot_y1, plot_x2, plot_y2]

            alpha_png = template[:, :, 3] / 255.0 * alpha
            alpha_png = np.expand_dims(alpha_png, axis=2)
            alpha_png = np.repeat(alpha_png, 3, axis=2)
            alpha_jpg = 1 - alpha_png

            img_result[iou[1]:iou[3], iou[0]:iou[2], :3] = ((alpha_jpg * img[iou[1]:iou[3], iou[0]:iou[2], :3]) +
                                                      (alpha_png * template[:, :, 2::-1]))
    return img_result


def paste_text_img(img, text_value, text_location, text_font):
    text_x = round(text_location[0] + (text_location[2] - text_location[0]) * 0.3)
    text_y = round(text_location[1] + (text_location[3] - text_location[1]) * 0.75)
    cv2.putText(img, str(text_value), (text_x, text_y), 0, text_font, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


def paste_text_img_long(img, text_value, text_location, text_font):
    text_x = round(text_location[0] + (text_location[2] - text_location[0]) * 0.05)
    text_y = round(text_location[1] + (text_location[3] - text_location[1]) * 0.75)
    cv2.putText(img, str(text_value), (text_x, text_y), 0, text_font, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


def paste_text_img_mid(img, text_value, text_location, text_font):
    text_x = round(text_location[0] + (text_location[2] - text_location[0]) * 0.15)
    text_y = round(text_location[1] + (text_location[3] - text_location[1]) * 0.75)
    cv2.putText(img, str(text_value), (text_x, text_y), 0, text_font, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


def add_bg(img, template):
    img_result = template.copy()
    for i in range(3):
        img_result[:,:,i][template[:,:,3]==0] = img[i]

    return img_result

def is_pos_in_bbox(pos, bbox):
    """
    :param pos: list [x, y]
    :param bbox: list [x1, y1, x2, y2], x1, y1 is the top left. x2, y2 is the bottom right
    :return: bool, True or False
    """
    if (pos[0] > bbox[0] and pos[0] < bbox[2]) and (pos[1] > bbox[1] and pos[1] < bbox[3]):
        return True
    else:
        return False


def is_multipos_in_bbox(multi_pos, bbox):
    """
    :param multi_pos: list [[x1, y1], [x2, y2], ...[xn, yn]]
    :param bbox: list [x1, y1, x2, y2], x1, y1 is the top left. x2, y2 is the bottom right
    :return: bool, True or False
    """
    multi_pos_array = np.array(multi_pos)
    # if position[0] < part_W * 2 and position[0] > 0
    x_exist_in_bbox = np.logical_and(multi_pos_array[:, 0] > bbox[0], multi_pos_array[:, 0] < bbox[2])
    y_exist_in_bbox = np.logical_and(multi_pos_array[:, 1] > bbox[1], multi_pos_array[:, 1] < bbox[3])
    exist_in_bbox = np.logical_and(x_exist_in_bbox, y_exist_in_bbox)
    return np.any(exist_in_bbox)


def is_multipos_in_bbox_with_idx(multi_pos, bbox):
    """
    :param multi_pos: list [[x1, y1], [x2, y2], ...[xn, yn]]
    :param bbox: list [x1, y1, x2, y2], x1, y1 is the top left. x2, y2 is the bottom right
    :return: bool, True or False
    """
    multi_pos_array = np.array(multi_pos)
    # if position[0] < part_W * 2 and position[0] > 0
    x_exist_in_bbox = np.logical_and(multi_pos_array[:, 0] > bbox[0], multi_pos_array[:, 0] < bbox[2])
    y_exist_in_bbox = np.logical_and(multi_pos_array[:, 1] > bbox[1], multi_pos_array[:, 1] < bbox[3])
    exist_in_bbox = np.logical_and(x_exist_in_bbox, y_exist_in_bbox)

    exist = np.any(exist_in_bbox)
    if exist:  # "if exist is True" is wrong since exist here is {bool_:()}, not the normal {bool}  type data || "if exist" or "if exist==True" both Okay
        idxs_2d = np.argwhere(exist_in_bbox == True)
        idxs_1d = np.squeeze(idxs_2d)
        # print('idx_1d: ', idxs_1d)
        idx = int(idxs_1d) if idxs_1d.size == 1 else np.random.choice(idxs_1d)
    else:
        idx = -1

    # print('exist_in_bbox: ', exist_in_bbox, 'exist: ', exist, 'idx: ', idx)

    return exist, idx


def RectErrorCal(pt, bbox):
    """
    Calculate the min distance from one point to a rect contour (outside the rect)
       Input:
       pt: the coordinates of a point (x, y)
       bbox: the rect points (top-left, bottom-right)
       Returns the error in x, y axis
    Here we seperate the area into 8 areas:
    1  |  2  | 3
    --------------
    4  |  0  | 5
    --------------
    6  /  7  / 8
    """
    pt_x = pt[0]
    pt_y = pt[1]
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    if (pt_x > 0 and pt_x < x1) and (pt_y > 0 and pt_y < y1):
        x_error = pt_x - x1
        y_error = pt_y - y1
    elif (pt_x > x1 and pt_x < x2) and (pt_y > 0 and pt_y < y1):
        x_error = 0
        y_error = pt_y - y1
    elif (pt_x > x2) and (pt_y > 0 and pt_y < y1):
        x_error = pt_x - x2
        y_error = pt_y - y1
    elif (pt_x > 0 and pt_x < x1) and (pt_y > y1 and pt_y < y2):
        x_error = pt_x - x1
        y_error = 0
    # elif (pt_x > x1 and pt_x < x2) and (pt_y > y1 and pt_y < y2):
    #     x_error = 0
    #     y_error = 0
    elif (pt_x > x2) and (pt_y > y1 and pt_y < y2):
        x_error = pt_x - x2
        y_error = 0
    elif (pt_x > 0 and pt_x < x1) and (pt_y > y2):
        x_error = pt_x - x1
        y_error = pt_y - y2
    elif (pt_x > x1 and pt_x < x2) and (pt_y > y2):
        x_error = 0
        y_error = pt_y - y2
    elif (pt_x > x2) and (pt_y > y2):
        x_error = pt_x - x2
        y_error = pt_y - y2
    else:
        x_error = 0
        y_error = 0

    return x_error, y_error


def DepthActionCal(depth, minmax):
    """
    Calculate the [-1, 1] depth action with the setting bound,
    """
    reference = (minmax[1] - minmax[0])
    if depth <= 0 or (depth > minmax[0] and depth < minmax[1]):
        return 0.0, 0.0
    else:
        if depth <= minmax[0]:
            return (depth - minmax[0]), (depth - minmax[0]) / reference
        if depth >= minmax[1]:
            return (depth - minmax[1]), (depth - minmax[1]) / reference
