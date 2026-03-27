import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class SwitchButton(QWidget):
    clicked = pyqtSignal()
    def __init__(self, parent=None):
        super(SwitchButton, self).__init__(parent)

        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.resize(70, 30)
        self.state = False

    def mousePressEvent(self, event):
        super(SwitchButton, self).mousePressEvent(event)

        self.state = False if self.state else True
        self.update()
        self.clicked.emit()

    def paintEvent(self, event):
        super(SwitchButton, self).paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        font = QFont('Microsoft YaHei')
        font.setPixelSize(14)
        painter.setFont(font)

        if self.state:
            painter.setPen(Qt.NoPen)
            brush = QBrush(QColor('#FF475D'))
            painter.setBrush(brush)
            painter.drawRoundedRect(0, 0, self.width(), self.height(), self.height() // 2, self.height() // 2)

            painter.setPen(Qt.NoPen)
            brush.setColor(QColor('#ffffff'))
            painter.setBrush(brush)
            painter.drawRoundedRect(43, 3, 24, 24, 12, 12)

            painter.setPen(QPen(QColor('#ffffff')))
            painter.setBrush(Qt.NoBrush)
            painter.drawText(QRect(18, 4, 50, 20), Qt.AlignLeft, 'On')
        else:
            painter.setPen(Qt.NoPen)
            brush = QBrush(QColor('#FFFFFF'))
            painter.setBrush(brush)
            painter.drawRoundedRect(0, 0, self.width(), self.height(), self.height()//2, self.height()//2)

            pen = QPen(QColor('#999999'))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawRoundedRect(3, 3, 24, 24, 12, 12)

            painter.setBrush(Qt.NoBrush)
            painter.drawText(QRect(38, 4, 50, 20), Qt.AlignLeft, 'Off')
