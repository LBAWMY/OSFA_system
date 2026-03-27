# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.label_img_show = QtWidgets.QLabel(MainWindow)
        self.label_img_show.setEnabled(True)
        self.label_img_show.setGeometry(QtCore.QRect(0, 0, 1440, 810))
        font = QtGui.QFont()
        font.setUnderline(False)
        font.setKerning(False)
        self.label_img_show.setFont(font)
        self.label_img_show.setAutoFillBackground(False)
        self.label_img_show.setFrameShape(QtWidgets.QFrame.Box)
        self.label_img_show.setText("")
        self.label_img_show.setObjectName("label_img_show")
        self.pushButton_start = QtWidgets.QPushButton(MainWindow)
        self.pushButton_start.setGeometry(QtCore.QRect(0, 810, 89, 25))
        self.pushButton_start.setObjectName("pushButton_start")
        self.pushButton_end = QtWidgets.QPushButton(MainWindow)
        self.pushButton_end.setGeometry(QtCore.QRect(90, 810, 89, 25))
        self.pushButton_end.setObjectName("pushButton_end")
        self.pushButton_control = QtWidgets.QPushButton(MainWindow)
        self.pushButton_control.setGeometry(QtCore.QRect(180, 810, 89, 25))
        self.pushButton_control.setObjectName("pushButton_control")

        self.retranslateUi(MainWindow)
        self.pushButton_end.clicked.connect(MainWindow.pushButton_end_clicked)
        self.pushButton_start.clicked.connect(MainWindow.pushButton_start_clicked)
        self.pushButton_control.clicked.connect(MainWindow.pushButton_Test_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Automatic Camera Control System"))
        self.pushButton_start.setText(_translate("MainWindow", "Start"))
        self.pushButton_end.setText(_translate("MainWindow", "End"))
        self.pushButton_control.setText(_translate("MainWindow", "TestJoyCtrl"))

