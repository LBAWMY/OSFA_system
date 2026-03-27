# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Right_1080p.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow

class Ui_Right1080p(QMainWindow):
    def setupUi(self, Right1080p):
        W = 1920
        H = 1080 - 27
        menu_bar_h = 22
        # W = 1280
        # H = 720
        # menu_bar_h = 10
        Right1080p.setObjectName("Right1080p")
        Right1080p.resize(W, H)
        self.centralwidget = QtWidgets.QWidget(Right1080p)
        self.centralwidget.setObjectName("centralwidget")
        # self.label = QtWidgets.QLabel(self.centralwidget)
        # self.label.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
        # self.label.setText("")
        # self.label.setObjectName("label")
        self.label_imgR_1080p_show = QtWidgets.QLabel(self.centralwidget)
        self.label_imgR_1080p_show.setEnabled(True)
        self.label_imgR_1080p_show.setGeometry(QtCore.QRect(0, 0, W, H))
        font = QtGui.QFont()
        font.setUnderline(False)
        font.setKerning(False)
        self.label_imgR_1080p_show.setFont(font)
        self.label_imgR_1080p_show.setAutoFillBackground(False)
        self.label_imgR_1080p_show.setFrameShape(QtWidgets.QFrame.Box)
        self.label_imgR_1080p_show.setText("")
        self.label_imgR_1080p_show.setObjectName("label_imgR_1080p_show")

        Right1080p.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Right1080p)
        self.menubar.setGeometry(QtCore.QRect(0, 0, W, menu_bar_h))
        self.menubar.setObjectName("menubar")
        Right1080p.setMenuBar(self.menubar)
        # self.statusbar = QtWidgets.QStatusBar(Right1080p)
        # self.statusbar.setObjectName("statusbar")
        # Right1080p.setStatusBar(self.statusbar)

        self.retranslateUi(Right1080p)
        QtCore.QMetaObject.connectSlotsByName(Right1080p)

    def retranslateUi(self, Right1080p):
        _translate = QtCore.QCoreApplication.translate
        Right1080p.setWindowTitle(_translate("Right1080p", "MainWindow"))

