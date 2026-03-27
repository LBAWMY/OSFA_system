# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_interface.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from qt_gui.Left_1080p import Ui_Left1080p
from qt_gui.Right_1080p import Ui_Right1080p
from qt_gui.QxtSpanSlider import QxtSpanSlider
from qt_gui.custom_joystick import CustomJoy
from qt_gui.LED import StatusLed
from qt_gui.SWITCH import SwitchButton
# from joystick import JoystickView, JoystickPointView
from qt_gui.qrangeslider import QRangeSlider
from PyQt5.QtGui import QPalette, QLinearGradient, QPen, QColor
from PyQt5.QtCore import (Qt, QCoreApplication,
                         QRect, QRectF, QPoint,
                         pyqtSignal as Signal, pyqtProperty as Property)
from PyQt5.QtWidgets import QSlider

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1859, 1080)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Reset_Depth = QtWidgets.QPushButton(self.centralwidget)
        self.Reset_Depth.setGeometry(QtCore.QRect(870, 90, 80, 50))
        font = QtGui.QFont()
        font.setPointSize(12) # 15
        font.setBold(True)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.Reset_Depth.setFont(font)
        self.Reset_Depth.setObjectName("Reset_Depth")
        self.Reset_PosY = QtWidgets.QPushButton(self.centralwidget)
        self.Reset_PosY.setGeometry(QtCore.QRect(720, 820, 80, 50))
        font = QtGui.QFont()
        font.setPointSize(12) # 15
        font.setBold(True)
        font.setWeight(75)
        self.Reset_PosY.setFont(font)
        self.Reset_PosY.setObjectName("Reset_PosY")
        self.Reset_PosX = QtWidgets.QPushButton(self.centralwidget)
        self.Reset_PosX.setGeometry(QtCore.QRect(940, 950, 81, 51))
        font = QtGui.QFont()
        font.setPointSize(12) # 15
        font.setBold(True)
        font.setWeight(75)
        self.Reset_PosX.setFont(font)
        self.Reset_PosX.setObjectName("Reset_PosX")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 590, 261, 61))
        font = QtGui.QFont()
        font.setPointSize(17) # 19
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_panel_img_show = QtWidgets.QLabel(self.centralwidget)
        self.label_panel_img_show.setEnabled(True)
        self.label_panel_img_show.setGeometry(QtCore.QRect(870, 160, 1024, 768))
        font = QtGui.QFont()
        font.setUnderline(False)
        font.setKerning(False)
        self.label_panel_img_show.setFont(font)
        self.label_panel_img_show.setAutoFillBackground(False)
        self.label_panel_img_show.setFrameShape(QtWidgets.QFrame.Box)
        self.label_panel_img_show.setText("")
        self.label_panel_img_show.setObjectName("label_panel_img_show")
        self.Show_L_1080p = QtWidgets.QPushButton(self.centralwidget)
        self.Show_L_1080p.setGeometry(QtCore.QRect(870, 950, 51, 51))
        self.Show_L_1080p.setObjectName("Show_L_1080p")
        self.Show_R_1080p = QtWidgets.QPushButton(self.centralwidget)
        self.Show_R_1080p.setGeometry(QtCore.QRect(1840, 950, 51, 51))
        self.Show_R_1080p.setObjectName("Show_R_1080p")
        self.PosX_Disp = QtWidgets.QLabel(self.centralwidget)
        self.PosX_Disp.setGeometry(QtCore.QRect(1330, 990, 120, 31)) # (1330, 990, 101, 31)
        self.PosX_Disp.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.PosX_Disp.setAutoFillBackground(False)
        self.PosX_Disp.setObjectName("PosX_Disp")
        self.PosY_Disp = QtWidgets.QLabel(self.centralwidget)
        self.PosY_Disp.setGeometry(QtCore.QRect(705, 530, 101, 31))
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.PosY_Disp.setFont(font)
        self.PosY_Disp.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.PosY_Disp.setAutoFillBackground(False)
        self.PosY_Disp.setAlignment(QtCore.Qt.AlignCenter)
        self.PosY_Disp.setObjectName("PosY_Disp")
        self.PosY_Disp_2 = QtWidgets.QLabel(self.centralwidget)
        self.PosY_Disp_2.setGeometry(QtCore.QRect(1320, 120, 100, 40)) # (1320, 120, 101, 31)
        self.PosY_Disp_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.PosY_Disp_2.setAutoFillBackground(False)
        self.PosY_Disp_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.PosY_Disp_2.setObjectName("PosY_Disp_2")
        self.MidDepth_Disp = QtWidgets.QLabel(self.centralwidget)
        self.MidDepth_Disp.setGeometry(QtCore.QRect(1280, 60, 120, 31)) # (1320, 120, 101, 31)
        self.MidDepth_Disp.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.MidDepth_Disp.setAutoFillBackground(False)
        self.MidDepth_Disp.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.MidDepth_Disp.setObjectName("MidDepth_Disp")
        self.lineEdit_14 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_14.setGeometry(QtCore.QRect(1400, 60, 100, 31))
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_14.setFont(font)
        self.lineEdit_14.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.right_mini = QtWidgets.QPushButton(self.centralwidget)
        self.right_mini.setGeometry(QtCore.QRect(1830, 20, 15, 15))
        self.right_mini.setObjectName("right_mini")
        self.right_visit = QtWidgets.QPushButton(self.centralwidget)
        self.right_visit.setGeometry(QtCore.QRect(1850, 20, 15, 15))
        self.right_visit.setText("")
        self.right_visit.setObjectName("right_visit")
        self.right_close = QtWidgets.QPushButton(self.centralwidget)
        self.right_close.setGeometry(QtCore.QRect(1870, 20, 15, 15))
        self.right_close.setText("")
        self.right_close.setObjectName("right_mini_3")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(50, 163, 601, 421))
        font = QtGui.QFont()
        font.setPointSize(12) # 15
        self.groupBox.setFont(font)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(20, 110, 71, 31))
        self.label_5.setObjectName("label_5")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(470, 50, 81, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit.setFont(font)
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 170, 91, 51))
        self.label_2.setObjectName("label_2")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(20, 250, 121, 31))
        self.label_4.setObjectName("label_4")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(320, 170, 51, 31))
        self.label_3.setObjectName("label_3")
        self.label_16 = QtWidgets.QLabel(self.groupBox)
        self.label_16.setGeometry(QtCore.QRect(20, 50, 161, 31))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.groupBox)
        self.label_17.setGeometry(QtCore.QRect(320, 50, 131, 31))
        self.label_17.setObjectName("label_17")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setGeometry(QtCore.QRect(120, 110, 113, 25))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_4.setGeometry(QtCore.QRect(472, 110, 81, 25))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_6.setGeometry(QtCore.QRect(120, 180, 31, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(320, 250, 141, 31))
        self.label_9.setObjectName("label_9")
        self.label_main_img_show = QtWidgets.QLabel(self.groupBox)
        self.label_main_img_show.setEnabled(True)
        self.label_main_img_show.setGeometry(QtCore.QRect(160, 150, 120, 80))
        font = QtGui.QFont()
        font.setUnderline(False)
        font.setKerning(False)
        self.label_main_img_show.setFont(font)
        self.label_main_img_show.setAutoFillBackground(False)
        self.label_main_img_show.setFrameShape(QtWidgets.QFrame.Box)
        self.label_main_img_show.setText("")
        self.label_main_img_show.setObjectName("label_main_img_show")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_5.setGeometry(QtCore.QRect(470, 180, 81, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.label_25 = QtWidgets.QLabel(self.groupBox)
        self.label_25.setGeometry(QtCore.QRect(320, 110, 71, 31))
        self.label_25.setObjectName("label_25")
        self.dial = QtWidgets.QDial(self.groupBox)
        self.dial.setGeometry(QtCore.QRect(440, 280, 131, 111))
        self.dial.setProperty("value", 50)
        self.dial.setSliderPosition(50)
        self.dial.setWrapping(True)
        self.dial.setNotchesVisible(True)
        self.dial.setObjectName("dial")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 130, 281, 61))
        font = QtGui.QFont()
        font.setPointSize(17) # 19
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(50, 625, 601, 391))
        font = QtGui.QFont()
        font.setPointSize(12) # 15
        self.groupBox_2.setFont(font)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(20, 210, 121, 31))
        self.label_7.setObjectName("label_7")
        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        self.label_18.setGeometry(QtCore.QRect(20, 50, 161, 31))
        self.label_18.setObjectName("label_18")
        self.lineEdit_7 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_7.setGeometry(QtCore.QRect(120, 110, 113, 25))
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.label_19 = QtWidgets.QLabel(self.groupBox_2)
        self.label_19.setGeometry(QtCore.QRect(320, 50, 131, 31))
        self.label_19.setObjectName("label_19")
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setGeometry(QtCore.QRect(20, 110, 71, 31))
        self.label_8.setObjectName("label_8")
        self.lineEdit_8 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_8.setGeometry(QtCore.QRect(470, 50, 81, 25))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lineEdit_8.setFont(font)
        self.lineEdit_8.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.lineEdit_9 = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_9.setGeometry(QtCore.QRect(120, 160, 113, 25))
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.label_27 = QtWidgets.QLabel(self.groupBox_2)
        self.label_27.setGeometry(QtCore.QRect(20, 160, 71, 31))
        self.label_27.setObjectName("label_27")

        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        font.setPointSize(12) # 15
        self.label_uterus_pitch = QtWidgets.QLabel(self.groupBox_2)
        self.label_uterus_pitch.setGeometry(QtCore.QRect(320, 110, 121, 31))
        self.label_uterus_pitch.setObjectName("label_uterus_pitch")
        self.lineEdit_uterus_pitch = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_uterus_pitch.setGeometry(QtCore.QRect(472, 110, 81, 31))
        self.lineEdit_uterus_pitch.setFont(font)
        self.lineEdit_uterus_pitch.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_uterus_pitch.setObjectName("lineEdit_uterus_pitch")

        self.label_uterus_yaw = QtWidgets.QLabel(self.groupBox_2)
        self.label_uterus_yaw.setGeometry(QtCore.QRect(320, 165, 121, 31))
        self.label_uterus_yaw.setObjectName("label_uterus_yaw")
        self.lineEdit_uterus_yaw = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_uterus_yaw.setGeometry(QtCore.QRect(472, 165, 81, 31))
        self.lineEdit_uterus_yaw.setFont(font)
        self.lineEdit_uterus_yaw.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_uterus_yaw.setObjectName("lineEdit_uterus_yaw")

        self.label_uterus_insertion = QtWidgets.QLabel(self.groupBox_2)
        self.label_uterus_insertion.setGeometry(QtCore.QRect(320, 220, 121, 31))
        self.label_uterus_insertion.setObjectName("label_uterus_insertion")
        self.lineEdit_uterus_insertion = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_uterus_insertion.setGeometry(QtCore.QRect(472, 220, 81, 31))
        self.lineEdit_uterus_insertion.setFont(font)
        self.lineEdit_uterus_insertion.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_uterus_insertion.setObjectName("lineEdit_uterus_insertion")

        self.label_uterus_rotation = QtWidgets.QLabel(self.groupBox_2)
        self.label_uterus_rotation.setGeometry(QtCore.QRect(320, 275, 121, 31))
        self.label_uterus_rotation.setObjectName("label_uterus_rotation")
        self.lineEdit_uterus_rotation = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_uterus_rotation.setGeometry(QtCore.QRect(472, 275, 81, 31))
        self.lineEdit_uterus_rotation.setFont(font)
        self.lineEdit_uterus_rotation.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_uterus_rotation.setObjectName("lineEdit_uterus_rotation")

        self.label_uterus_grasp = QtWidgets.QLabel(self.groupBox_2)
        self.label_uterus_grasp.setGeometry(QtCore.QRect(320, 330, 121, 31))
        self.label_uterus_grasp.setObjectName("label_uterus_grasp")
        self.lineEdit_uterus_grasp = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit_uterus_grasp.setGeometry(QtCore.QRect(472, 330, 81, 31))
        self.lineEdit_uterus_grasp.setFont(font)
        self.lineEdit_uterus_grasp.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_uterus_grasp.setObjectName("lineEdit_uterus_grasp")

        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(20, 0, 301, 61))
        font = QtGui.QFont()
        font.setPointSize(17) # 19
        font.setBold(True)
        font.setWeight(75)

        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(50, 35, 601, 91))
        font = QtGui.QFont()
        font.setPointSize(12) # 15
        self.groupBox_3.setFont(font)
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_11 = QtWidgets.QLabel(self.groupBox_3)
        self.label_11.setGeometry(QtCore.QRect(20, 50, 101, 31))
        self.label_11.setObjectName("label_11")
        self.label_20 = QtWidgets.QLabel(self.groupBox_3)
        self.label_20.setGeometry(QtCore.QRect(310, 50, 71, 31))
        self.label_20.setObjectName("label_20")
        self.lineEdit_10 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_10.setGeometry(QtCore.QRect(410, 50, 113, 25))
        self.lineEdit_10.setObjectName("lineEdit_10")

        self.lineEdit_11 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_11.setGeometry(QtCore.QRect(715, 900, 91, 41)) # (710, 180, 91, 41)
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_11.setFont(font)
        self.lineEdit_11.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(715, 940, 91, 31)) # (710, 220, 91, 31)
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_21.setFont(font)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(1700, 1000, 91, 31))

        self.lineEdit_Version = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_Version.setGeometry(QtCore.QRect(710, 180, 91, 31)) # (710, 90, 91, 41)
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_Version.setFont(font)
        self.lineEdit_Version.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_Version.setObjectName("lineEdit_Version")
        self.label_version = QtWidgets.QLabel(self.centralwidget)
        self.label_version.setGeometry(QtCore.QRect(695, 210, 120, 31)) # (710, 220, 91, 31)
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_version.setFont(font)
        self.label_version.setAlignment(QtCore.Qt.AlignCenter)
        self.label_version.setObjectName("label_version")

        self.lineEdit_InitialDepth = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_InitialDepth.setGeometry(QtCore.QRect(710, 260, 91, 31)) # (710, 90, 91, 41)
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_InitialDepth.setFont(font)
        self.lineEdit_InitialDepth.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_InitialDepth.setObjectName("lineEdit_InitialDepth")
        self.label_InitialDepth = QtWidgets.QLabel(self.centralwidget)
        self.label_InitialDepth.setGeometry(QtCore.QRect(695, 290, 120, 31)) # (710, 220, 91, 31)
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_InitialDepth.setFont(font)
        self.label_InitialDepth.setAlignment(QtCore.Qt.AlignCenter)
        self.label_InitialDepth.setObjectName("label_InitialDepth")

        self.lineEdit_WorkspaceRange = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_WorkspaceRange.setGeometry(QtCore.QRect(710, 340, 91, 31)) # (710, 90, 91, 41)
        font = QtGui.QFont()
        font.setPointSize(12) # 18
        self.lineEdit_WorkspaceRange.setFont(font)
        self.lineEdit_WorkspaceRange.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_WorkspaceRange.setObjectName("lineEdit_WorkspaceRange")
        self.label_workspace = QtWidgets.QLabel(self.centralwidget)
        self.label_workspace.setGeometry(QtCore.QRect(690, 370, 130, 31)) # (710, 220, 91, 31)
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_workspace.setFont(font)
        self.label_workspace.setAlignment(QtCore.Qt.AlignCenter)
        self.label_workspace.setObjectName("label_WorkspaceRange")

        self.lineEdit_RCM2CAM = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_RCM2CAM.setGeometry(QtCore.QRect(710, 420, 91, 31)) # (710, 90, 91, 41)
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_RCM2CAM.setFont(font)
        self.lineEdit_RCM2CAM.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_RCM2CAM.setObjectName("lineEdit_RCM2CAM")
        self.label_rcm2cam = QtWidgets.QLabel(self.centralwidget)
        self.label_rcm2cam.setGeometry(QtCore.QRect(695, 450, 120, 31)) # (710, 220, 91, 31)
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_rcm2cam.setFont(font)
        self.label_rcm2cam.setAlignment(QtCore.Qt.AlignCenter)
        self.label_rcm2cam.setObjectName("label_rcm2cam")

        self.lineEdit_SurgicalPhase = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_SurgicalPhase.setGeometry(QtCore.QRect(710, 610, 91, 31)) # (710, 90, 91, 41)
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_SurgicalPhase.setFont(font)
        self.lineEdit_SurgicalPhase.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_SurgicalPhase.setObjectName("lineEdit_SurgicalPhase")
        self.label_surgicalphase = QtWidgets.QLabel(self.centralwidget)
        self.label_surgicalphase.setGeometry(QtCore.QRect(695, 640, 120, 31)) # (710, 220, 91, 31)
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_surgicalphase.setFont(font)
        self.label_surgicalphase.setAlignment(QtCore.Qt.AlignCenter)
        self.label_surgicalphase.setObjectName("label_surgicalphase")

        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_22.setFont(font)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.lineEdit_12 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_12.setGeometry(QtCore.QRect(1700, 955, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_12.setFont(font)
        self.lineEdit_12.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(1800, 120, 91, 31))
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_23.setFont(font)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.lineEdit_13 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_13.setGeometry(QtCore.QRect(1800, 80, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(15) # 18
        self.lineEdit_13.setFont(font)
        self.lineEdit_13.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_13.setObjectName("lineEdit_13")
        # self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        # self.verticalSlider.setGeometry(QtCore.QRect(840, 160, 21, 771))
        # self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        # self.verticalSlider.setObjectName("verticalSlider")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(670, 50, 20, 961))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(850, 1000, 91, 31))
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_24.setFont(font)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(1820, 1000, 100, 31))
        font = QtGui.QFont()
        font.setItalic(True)
        font.setUnderline(True)
        self.label_26.setFont(font)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(870, 930, 1021, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(850, 160, 20, 771))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1859, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        #################################################### add the joystick + slider
        self.joystick_Laparoscope = CustomJoy(self.groupBox)
        self.joystick_Laparoscope.setGeometry(QtCore.QRect(55, 275, 150, 150))
        # self.joystick_Laparoscope.setPosXY(0, 0)
        self.slider_Laparoscope = QSlider(self.groupBox)
        self.slider_Laparoscope.setGeometry(QtCore.QRect(220, 290, 20, 120)) #4FA6EB; 7A7A7A
        self.slider_Laparoscope.setStyleSheet('''
        QSlider 
        {
            background-color: rgba(220, 220, 220, 0.7);
            padding-top: 15px;
            padding-bottom: 15px;
            border-radius: 5px;
        }
        QSlider::add-page:vertical 
        {
            background-color: #4FA6EB;
            width:5px;
            border-radius: 2px;
        }
        
        QSlider::sub-page:vertical 
        {
            background-color: #FFA500;
            width:5px;
            border-radius: 2px;
        }
        
        QSlider::groove:vertical 
        {
            background:transparent;
            width:6px;
        }
        
        QSlider::handle:vertical    
        {
            height: 14px;  
            width: 14px;
            margin: 0px -4px 0px -4px;
            border-radius: 7px;
            background: white;
        }
        ''')
        self.slider_Laparoscope.setSliderPosition(50)
        self.joystick_Uterus_manipulator = CustomJoy(self.groupBox_2)
        self.joystick_Uterus_manipulator.setGeometry(QtCore.QRect(55, 235, 150, 150))
        # self.joystick_Uterus_manipulator.setPosXY(0, 0)
        self.slider_Uterus_manipulator = QSlider(self.groupBox_2)
        self.slider_Uterus_manipulator.setGeometry(QtCore.QRect(220, 250, 20, 120))
        self.slider_Uterus_manipulator.setStyleSheet('''
        QSlider 
        {
            background-color: rgba(220, 220, 220, 0.7);
            padding-top: 15px;
            padding-bottom: 15px;
            border-radius: 5px;
        }
        QSlider::add-page:vertical 
        {
            background-color: #4FA6EB;
            width:5px;
            border-radius: 2px;
        }
        
        QSlider::sub-page:vertical 
        {
            background-color: #FFA500;
            width:5px;
            border-radius: 2px;
        }
        
        QSlider::groove:vertical 
        {
            background:transparent;
            width:6px;
        }
        
        QSlider::handle:vertical    
        {
            height: 14px;  
            width: 14px;
            margin: 0px -4px 0px -4px;
            border-radius: 7px;
            background: white;
        }
        ''')
        self.slider_Uterus_manipulator.setSliderPosition(50)

        #################################################### add the bar
        # self.spanslider_x = QRangeSlider(self.centralwidget)
        # self.spanslider_x.setGeometry(QtCore.QRect(870, 930, 1021, 10))
        # # self.spanslider_x.setSpan(30, 70)
        # self.spanslider_x.setRange(0, 100)
        # # self.spanslider_x.setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222, stop:1 #333);')
        # # self.spanslider_x.handle.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282, stop:1 #393);')
        # self.spanslider_x.setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #aba7a7, stop:1 #aba7a7);')
        # self.spanslider_x.handle.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #eb7171, stop:1 #f20a0a);')
        self.spanslider_x = QxtSpanSlider(self.centralwidget)
        self.spanslider_x.setGeometry(QtCore.QRect(870, 930, 1021, 10))
        self.spanslider_x.setSpan(0, 100)
        color = QColor(Qt.red).lighter(80)
        self.spanslider_x.setGradientLeftColor(color)
        self.spanslider_x.setGradientRightColor(color)

        self.spanslider_y = QxtSpanSlider(self.centralwidget)
        self.spanslider_y.setGeometry(QtCore.QRect(840, 160, 21, 771))
        self.spanslider_y.setOrientation(QtCore.Qt.Vertical)
        self.spanslider_y.setSpan(0, 100)
        color = QColor(Qt.green).lighter(80)
        self.spanslider_y.setGradientLeftColor(color)
        self.spanslider_y.setGradientRightColor(color)
        # self.spanslider_y.setRange(30, 90)

        # self.spanslider_z = QRangeSlider(self.centralwidget)
        # self.spanslider_z.setGeometry(QtCore.QRect(1000, 100, 755, 10))
        # self.spanslider_z.setMax(120)
        # # self.spanslider_z.setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222, stop:1 #333);')
        # # self.spanslider_z.handle.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282, stop:1 #393);')
        # self.spanslider_z.setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #aba7a7, stop:1 #aba7a7);')
        # self.spanslider_z.handle.setStyleSheet('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #716fe8, stop:1 #250af2);')
        self.spanslider_z = QxtSpanSlider(self.centralwidget)
        self.spanslider_z.setGeometry(QtCore.QRect(1000, 100, 755, 10))
        self.spanslider_z.setMaximum(120)
        color = QColor(Qt.blue).lighter(80)
        self.spanslider_z.setGradientLeftColor(color)
        self.spanslider_z.setGradientRightColor(color)

        # self.PosY_Disp.setOrientation(QtCore.Qt.Vertical)
        # add the LED
        self.statusled_GUI = StatusLed(self.groupBox_3)
        self.statusled_GUI.setGeometry(QtCore.QRect(200, 50, 31, 31))
        # self.statusled_GUI.setCheckable(False)
        # self.statusled_GUI.click() # click once to change status, more click to convert the state back.
        self.statusled_LaparoscopeHolder = StatusLed(self.groupBox)
        self.statusled_LaparoscopeHolder.setGeometry(QtCore.QRect(200, 50, 31, 31))
        self.statusled_LaparoscopeHolder.click()
        self.statusled_UterusManipulator = StatusLed(self.groupBox_2)
        self.statusled_UterusManipulator.setGeometry(QtCore.QRect(200, 50, 31, 31))
        self.statusled_UterusManipulator.click()

        # add SWITCH button
        self.switchbutton_system = SwitchButton(self.centralwidget)
        self.switchbutton_system.setGeometry(QtCore.QRect(710, 90, 70, 30)) # (725, 900, 70, 30)

        # add the joystick
        # self.joy_LaparoscopeHolder = JoystickView(self.groupBox)
        # self.joy_LaparoscopeHolder.setGeometry(QtCore.QRect(200, 280, 131, 131))
        # self.laparoscope_action_x = QtWidgets.QFrame(self.groupBox)
        # self.laparoscope_action_x.setGeometry(QtCore.QRect(70, 350, 118, 3))
        # self.laparoscope_action_x.setFrameShape(QtWidgets.QFrame.HLine)
        # self.laparoscope_action_x.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.laparoscope_action_x.setObjectName("laparoscope_action_x")
        #
        # self.laparoscope_action_y = QtWidgets.QFrame(self.groupBox)
        # self.laparoscope_action_y.setGeometry(QtCore.QRect(120, 290, 20, 121))
        # self.laparoscope_action_y.setFrameShape(QtWidgets.QFrame.VLine)
        # self.laparoscope_action_y.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.laparoscope_action_y.setObjectName("laparoscope_action_y")
        #
        # self.laparoscope_action_z = QtWidgets.QFrame(self.groupBox)
        # self.laparoscope_action_z.setGeometry(QtCore.QRect(220, 290, 20, 121))
        # self.laparoscope_action_z.setFrameShape(QtWidgets.QFrame.VLine)
        # self.laparoscope_action_z.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.laparoscope_action_z.setObjectName("laparoscope_action_z")

        # self.uterus_action_x = QtWidgets.QFrame(self.groupBox_2)
        # self.uterus_action_x.setGeometry(QtCore.QRect(60, 270, 118, 3))
        # self.uterus_action_x.setFrameShape(QtWidgets.QFrame.HLine)
        # self.uterus_action_x.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.uterus_action_x.setObjectName("uterus_action_x")
        #
        # self.uterus_action_y = QtWidgets.QFrame(self.groupBox_2)
        # self.uterus_action_y.setGeometry(QtCore.QRect(110, 210, 20, 121))
        # self.uterus_action_y.setFrameShape(QtWidgets.QFrame.VLine)
        # self.uterus_action_y.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.uterus_action_y.setObjectName("uterus_action_y")
        #
        # self.uterus_action_z = QtWidgets.QFrame(self.groupBox_2)
        # self.uterus_action_z.setGeometry(QtCore.QRect(210, 210, 20, 121))
        # self.uterus_action_z.setFrameShape(QtWidgets.QFrame.VLine)
        # self.uterus_action_z.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.uterus_action_z.setObjectName("uterus_action_z")

        # self.line_3 = QtWidgets.QFrame(self.centralwidget)
        # self.line_3.setGeometry(QtCore.QRect(850, 160, 20, 771))
        # self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        # self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        # self.line_3.setObjectName("line_3")

        self.right_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.right_visit.setStyleSheet('''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.right_mini.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')
        # MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.right_close.clicked.connect(QCoreApplication.instance().quit)
        # self.right_mini.clicked.connect(QCoreApplication.instance().mini)

        self.retranslateUi(MainWindow)
        # self.Show_L_1080p.clicked.connect(self.Show_L_Img_1080p) # added
        # self.Show_R_1080p.clicked.connect(self.Show_R_Img_1080p) # added
        # self.switchbutton_system.clicked.connect(self.Turn_On_Off_system)
        self.Show_L_1080p.clicked.connect(MainWindow.Show_L_Img_1080p_trigger) # added
        self.Show_R_1080p.clicked.connect(MainWindow.Show_R_Img_1080p_trigger) # added
        self.switchbutton_system.clicked.connect(MainWindow.Turn_On_Off_system)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Reset_Depth.setText(_translate("MainWindow", "Reset Z"))
        self.Reset_PosY.setText(_translate("MainWindow", "Reset Y"))
        self.Reset_PosX.setText(_translate("MainWindow", "Reset X"))
        self.label_6.setText(_translate("MainWindow", "Uterus Manipulator"))
        self.Show_L_1080p.setText(_translate("MainWindow", "L"))
        self.Show_R_1080p.setText(_translate("MainWindow", "R"))
        self.PosX_Disp.setText(_translate("MainWindow", "<== X ==>"))
        self.PosY_Disp.setText(_translate("MainWindow", "<== Y ==>"))
        self.PosY_Disp_2.setText(_translate("MainWindow", "<== Z ==>"))
        self.MidDepth_Disp.setText(_translate("MainWindow", "Depth Around: "))
        self.right_mini.setText(_translate("MainWindow", " "))
        self.label_5.setText(_translate("MainWindow", "Mode"))
        self.lineEdit.setText(_translate("MainWindow", "0.00"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Main <br/>Instrument</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "Action Panel"))
        self.label_3.setText(_translate("MainWindow", "Type"))
        self.label_16.setText(_translate("MainWindow", "Emergency STOP"))
        self.label_17.setText(_translate("MainWindow", "Control Speed"))
        self.lineEdit_3.setText(_translate("MainWindow", "Manual/GUI/Automatic"))
        self.lineEdit_4.setText(_translate("MainWindow", "zoom in/out"))
        self.lineEdit_6.setText(_translate("MainWindow", "1"))
        self.label_9.setText(_translate("MainWindow", "Misorientation"))
        self.lineEdit_5.setText(_translate("MainWindow", "0.00"))
        self.label_25.setText(_translate("MainWindow", "For GUI"))
        self.label.setText(_translate("MainWindow", "Laparoscope Holder"))
        self.label_7.setText(_translate("MainWindow", "Action Panel"))
        self.label_18.setText(_translate("MainWindow", "Emergency STOP"))
        self.lineEdit_7.setText(_translate("MainWindow", "Manual/GUI/Automatic"))
        self.label_19.setText(_translate("MainWindow", "Control Speed"))
        self.label_8.setText(_translate("MainWindow", "Mode"))
        self.lineEdit_8.setText(_translate("MainWindow", "0.00"))
        self.lineEdit_9.setText(_translate("MainWindow", "zoom in/out"))
        self.label_27.setText(_translate("MainWindow", "For GUI"))

        self.label_10.setText(_translate("MainWindow", "GUI Interface Monitor"))
        self.label_11.setText(_translate("MainWindow", "Activated"))
        self.label_20.setText(_translate("MainWindow", "Status"))
        self.lineEdit_10.setText(_translate("MainWindow", "Uterus/Main/Settings/Camera/None"))
        self.lineEdit_11.setText(_translate("MainWindow", "0.00"))
        self.label_21.setText(_translate("MainWindow", "Pos Y info"))
        self.label_22.setText(_translate("MainWindow", "Pos X info"))
        self.lineEdit_12.setText(_translate("MainWindow", "0.00"))
        self.label_23.setText(_translate("MainWindow", "Depth info"))
        self.lineEdit_13.setText(_translate("MainWindow", "0.00"))
        self.lineEdit_14.setText(_translate("MainWindow", "0.00"))
        self.label_24.setText(_translate("MainWindow", "Left-1080p"))
        self.label_26.setText(_translate("MainWindow", "Right-1080p"))

        self.label_version.setText(_translate("MainWindow", "System Info"))
        self.label_InitialDepth.setText(_translate("MainWindow", "Init Workspace"))
        self.label_workspace.setText(_translate("MainWindow", "WS Range"))
        self.label_rcm2cam.setText(_translate("MainWindow", "Rcm2Cam"))
        self.label_surgicalphase.setText(_translate("MainWindow", "Surgical Phase"))

        self.label_uterus_pitch.setText(_translate("MainWindow", "Pitch"))
        self.label_uterus_yaw.setText(_translate("MainWindow", "Yaw"))
        self.label_uterus_insertion.setText(_translate("MainWindow", "Insertion"))
        self.label_uterus_rotation.setText(_translate("MainWindow", "Tilt"))
        self.label_uterus_grasp.setText(_translate("MainWindow", "Grasp"))
        # self.lineEdit_14.setText(_translate("MainWindow", "0.00"))

    # def Show_L_Img_1080p(self):
    #     self.left_1080p = Ui_Left1080p()
    #     self.left_1080p.show()
    #
    # def Show_R_Img_1080p(self):
    #     self.right_1080p = Ui_Right1080p()
    #     self.right_1080p.show()
    #
    # def Turn_On_Off_system(self):
    #     pass
        # self.MainProgram = MainWindow()
        # if self.switchbutton_system.status is True:
        #     self.MainProgram.system_on()
        # else:
        #     self.MainProgram.system_off()
