# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Left_1080p.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow

class Ui_Left1080p(QMainWindow):
    def setupUi(self, Left1080p):
        Left1080p.setObjectName("Left1080p")
        Left1080p.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(Left1080p)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
        self.widget.setObjectName("widget")
        Left1080p.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Left1080p)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 22))
        self.menubar.setObjectName("menubar")
        Left1080p.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Left1080p)
        self.statusbar.setObjectName("statusbar")
        Left1080p.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(Left1080p)
        self.toolBar.setObjectName("toolBar")
        Left1080p.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.toolBar_2 = QtWidgets.QToolBar(Left1080p)
        self.toolBar_2.setObjectName("toolBar_2")
        Left1080p.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar_2)

        self.retranslateUi(Left1080p)
        QtCore.QMetaObject.connectSlotsByName(Left1080p)

    def retranslateUi(self, Left1080p):
        _translate = QtCore.QCoreApplication.translate
        Left1080p.setWindowTitle(_translate("Left1080p", "MainWindow"))
        self.toolBar.setWindowTitle(_translate("Left1080p", "toolBar"))
        self.toolBar_2.setWindowTitle(_translate("Left1080p", "toolBar_2"))

