# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'iu.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(764, 448)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 40, 171, 321))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(30, 40, 111, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 90, 111, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(30, 210, 111, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(30, 260, 111, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 130, 131, 80))
        self.groupBox_5.setObjectName("groupBox_5")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 50, 111, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_5)
        self.lineEdit_2.setGeometry(QtCore.QRect(80, 20, 41, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setGeometry(QtCore.QRect(10, 20, 71, 20))
        self.label.setObjectName("label")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(570, 40, 171, 321))
        self.groupBox_4.setObjectName("groupBox_4")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_9.setGeometry(QtCore.QRect(10, 70, 151, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_10.setGeometry(QtCore.QRect(10, 140, 151, 31))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_11.setGeometry(QtCore.QRect(10, 210, 151, 31))
        self.pushButton_11.setObjectName("pushButton_11")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(390, 40, 171, 321))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 140, 151, 31))
        self.pushButton_8.setObjectName("pushButton_8")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(210, 40, 171, 321))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 140, 151, 31))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_7.setGeometry(QtCore.QRect(10, 210, 151, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(10, 70, 151, 31))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Calibration"))
        self.pushButton.setText(_translate("MainWindow", "1.1Find Corners"))
        self.pushButton_2.setText(_translate("MainWindow", "1.2Find Intrinsic"))
        self.pushButton_4.setText(_translate("MainWindow", "1.4Find Distortion"))
        self.pushButton_5.setText(_translate("MainWindow", "1.5Show Result"))
        self.groupBox_5.setTitle(_translate("MainWindow", "1.3Find Extrinsic"))
        self.pushButton_3.setText(_translate("MainWindow", "1.3Find Extrinsic"))
        self.label.setText(_translate("MainWindow", "Select images:"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. SIFT"))
        self.pushButton_9.setText(_translate("MainWindow", "4.1Keypoints"))
        self.pushButton_10.setText(_translate("MainWindow", "4.2Matched Keypoints"))
        self.pushButton_11.setText(_translate("MainWindow", "4.3Warp Images"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3.Stereo Disparity Map"))
        self.pushButton_8.setText(_translate("MainWindow", "3.1Stereo Disparity Map"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Argumented Reality"))
        self.pushButton_6.setText(_translate("MainWindow", "2.1Show Words on Board"))
        self.pushButton_7.setText(_translate("MainWindow", "2.2Show Words Vertically"))
