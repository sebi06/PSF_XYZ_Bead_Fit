# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\M1SRH\Documents\Spyder_Projects\PSF_XYZ_Bead_Fit_BioFormats\PSF_XYZ_Dialog_BF.ui'
#
# Created: Mon Jun 08 11:11:58 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_PSF_XYZ_Dialog_BF(object):
    def setupUi(self, PSF_XYZ_Dialog_BF):
        PSF_XYZ_Dialog_BF.setObjectName(_fromUtf8("PSF_XYZ_Dialog_BF"))
        PSF_XYZ_Dialog_BF.resize(731, 300)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PSF_XYZ_Dialog_BF.sizePolicy().hasHeightForWidth())
        PSF_XYZ_Dialog_BF.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        PSF_XYZ_Dialog_BF.setFont(font)
        self.layoutWidget = QtGui.QWidget(PSF_XYZ_Dialog_BF)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 712, 281))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setHorizontalSpacing(6)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.checkBox_SavePeaks = QtGui.QCheckBox(self.layoutWidget)
        self.checkBox_SavePeaks.setObjectName(_fromUtf8("checkBox_SavePeaks"))
        self.gridLayout.addWidget(self.checkBox_SavePeaks, 12, 0, 1, 1)
        self.label_channel = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_channel.setFont(font)
        self.label_channel.setObjectName(_fromUtf8("label_channel"))
        self.gridLayout.addWidget(self.label_channel, 3, 0, 1, 1)
        self.label_objNA = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_objNA.setFont(font)
        self.label_objNA.setObjectName(_fromUtf8("label_objNA"))
        self.gridLayout.addWidget(self.label_objNA, 2, 3, 1, 1)
        self.SpinBox_channel = QtGui.QSpinBox(self.layoutWidget)
        self.SpinBox_channel.setEnabled(True)
        self.SpinBox_channel.setMinimum(1)
        self.SpinBox_channel.setMaximum(6)
        self.SpinBox_channel.setObjectName(_fromUtf8("SpinBox_channel"))
        self.gridLayout.addWidget(self.SpinBox_channel, 3, 1, 1, 1)
        self.ExEm_text = QtGui.QLabel(self.layoutWidget)
        self.ExEm_text.setAlignment(QtCore.Qt.AlignCenter)
        self.ExEm_text.setObjectName(_fromUtf8("ExEm_text"))
        self.gridLayout.addWidget(self.ExEm_text, 4, 1, 1, 1)
        self.line_4 = QtGui.QFrame(self.layoutWidget)
        self.line_4.setFrameShape(QtGui.QFrame.HLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.gridLayout.addWidget(self.line_4, 5, 0, 1, 6)
        self.SpinBox_NA = QtGui.QDoubleSpinBox(self.layoutWidget)
        self.SpinBox_NA.setMinimum(0.0)
        self.SpinBox_NA.setMaximum(1.7)
        self.SpinBox_NA.setSingleStep(0.01)
        self.SpinBox_NA.setProperty("value", 0.0)
        self.SpinBox_NA.setObjectName(_fromUtf8("SpinBox_NA"))
        self.gridLayout.addWidget(self.SpinBox_NA, 2, 4, 1, 1)
        self.text_filename = QtGui.QLineEdit(self.layoutWidget)
        self.text_filename.setEnabled(True)
        self.text_filename.setObjectName(_fromUtf8("text_filename"))
        self.gridLayout.addWidget(self.text_filename, 0, 0, 1, 5)
        self.OpenFile = QtGui.QPushButton(self.layoutWidget)
        self.OpenFile.setObjectName(_fromUtf8("OpenFile"))
        self.gridLayout.addWidget(self.OpenFile, 0, 5, 1, 1)
        self.label_objname = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_objname.setFont(font)
        self.label_objname.setObjectName(_fromUtf8("label_objname"))
        self.gridLayout.addWidget(self.label_objname, 2, 0, 1, 1)
        self.label_threshold = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_threshold.setFont(font)
        self.label_threshold.setObjectName(_fromUtf8("label_threshold"))
        self.gridLayout.addWidget(self.label_threshold, 6, 0, 1, 1)
        self.SpinBox_threshold = QtGui.QDoubleSpinBox(self.layoutWidget)
        self.SpinBox_threshold.setToolTip(_fromUtf8(""))
        self.SpinBox_threshold.setMaximum(1.0)
        self.SpinBox_threshold.setSingleStep(0.01)
        self.SpinBox_threshold.setProperty("value", 0.5)
        self.SpinBox_threshold.setObjectName(_fromUtf8("SpinBox_threshold"))
        self.gridLayout.addWidget(self.SpinBox_threshold, 6, 1, 1, 1)
        self.label_pixsize = QtGui.QLabel(self.layoutWidget)
        self.label_pixsize.setObjectName(_fromUtf8("label_pixsize"))
        self.gridLayout.addWidget(self.label_pixsize, 7, 0, 1, 1)
        self.SpinBox_pixsize = QtGui.QDoubleSpinBox(self.layoutWidget)
        self.SpinBox_pixsize.setEnabled(True)
        self.SpinBox_pixsize.setToolTip(_fromUtf8(""))
        self.SpinBox_pixsize.setDecimals(1)
        self.SpinBox_pixsize.setMinimum(0.0)
        self.SpinBox_pixsize.setMaximum(1000.0)
        self.SpinBox_pixsize.setSingleStep(1.0)
        self.SpinBox_pixsize.setProperty("value", 0.0)
        self.SpinBox_pixsize.setObjectName(_fromUtf8("SpinBox_pixsize"))
        self.gridLayout.addWidget(self.SpinBox_pixsize, 7, 1, 1, 1)
        self.zspacing = QtGui.QLabel(self.layoutWidget)
        self.zspacing.setObjectName(_fromUtf8("zspacing"))
        self.gridLayout.addWidget(self.zspacing, 7, 2, 1, 1)
        self.label_guess_fwhmz = QtGui.QLabel(self.layoutWidget)
        self.label_guess_fwhmz.setObjectName(_fromUtf8("label_guess_fwhmz"))
        self.gridLayout.addWidget(self.label_guess_fwhmz, 8, 0, 1, 1)
        self.SpinBox_guess_fwhmz = QtGui.QDoubleSpinBox(self.layoutWidget)
        self.SpinBox_guess_fwhmz.setToolTip(_fromUtf8(""))
        self.SpinBox_guess_fwhmz.setDecimals(3)
        self.SpinBox_guess_fwhmz.setMinimum(0.1)
        self.SpinBox_guess_fwhmz.setMaximum(5.0)
        self.SpinBox_guess_fwhmz.setSingleStep(0.1)
        self.SpinBox_guess_fwhmz.setProperty("value", 1.0)
        self.SpinBox_guess_fwhmz.setObjectName(_fromUtf8("SpinBox_guess_fwhmz"))
        self.gridLayout.addWidget(self.SpinBox_guess_fwhmz, 8, 1, 1, 1)
        self.SpinBox_zspacing = QtGui.QDoubleSpinBox(self.layoutWidget)
        self.SpinBox_zspacing.setEnabled(True)
        self.SpinBox_zspacing.setMinimum(0.0)
        self.SpinBox_zspacing.setMaximum(5.0)
        self.SpinBox_zspacing.setSingleStep(0.0025)
        self.SpinBox_zspacing.setProperty("value", 0.0)
        self.SpinBox_zspacing.setObjectName(_fromUtf8("SpinBox_zspacing"))
        self.gridLayout.addWidget(self.SpinBox_zspacing, 7, 4, 1, 1)
        self.writexls = QtGui.QCheckBox(self.layoutWidget)
        self.writexls.setEnabled(True)
        self.writexls.setObjectName(_fromUtf8("writexls"))
        self.gridLayout.addWidget(self.writexls, 11, 0, 1, 1)
        self.line_3 = QtGui.QFrame(self.layoutWidget)
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.gridLayout.addWidget(self.line_3, 1, 0, 1, 6)
        self.label_ExEm = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_ExEm.setFont(font)
        self.label_ExEm.setObjectName(_fromUtf8("label_ExEm"))
        self.gridLayout.addWidget(self.label_ExEm, 4, 0, 1, 1)
        self.objname_text = QtGui.QLabel(self.layoutWidget)
        self.objname_text.setAlignment(QtCore.Qt.AlignCenter)
        self.objname_text.setObjectName(_fromUtf8("objname_text"))
        self.gridLayout.addWidget(self.objname_text, 2, 1, 1, 1)
        self.label_subimage_size = QtGui.QLabel(self.layoutWidget)
        self.label_subimage_size.setObjectName(_fromUtf8("label_subimage_size"))
        self.gridLayout.addWidget(self.label_subimage_size, 6, 2, 1, 1)
        self.SpinBox_subimage_size = QtGui.QSpinBox(self.layoutWidget)
        self.SpinBox_subimage_size.setMinimum(5)
        self.SpinBox_subimage_size.setMaximum(30)
        self.SpinBox_subimage_size.setProperty("value", 12)
        self.SpinBox_subimage_size.setObjectName(_fromUtf8("SpinBox_subimage_size"))
        self.gridLayout.addWidget(self.SpinBox_subimage_size, 6, 4, 1, 1)
        self.label_offset = QtGui.QLabel(self.layoutWidget)
        self.label_offset.setObjectName(_fromUtf8("label_offset"))
        self.gridLayout.addWidget(self.label_offset, 8, 2, 1, 1)
        self.SpinBox_offset = QtGui.QSpinBox(self.layoutWidget)
        self.SpinBox_offset.setMinimum(0)
        self.SpinBox_offset.setMaximum(50000)
        self.SpinBox_offset.setProperty("value", 0)
        self.SpinBox_offset.setObjectName(_fromUtf8("SpinBox_offset"))
        self.gridLayout.addWidget(self.SpinBox_offset, 8, 4, 1, 1)
        self.checkBox_SaveOrthoView = QtGui.QCheckBox(self.layoutWidget)
        self.checkBox_SaveOrthoView.setObjectName(_fromUtf8("checkBox_SaveOrthoView"))
        self.gridLayout.addWidget(self.checkBox_SaveOrthoView, 11, 2, 1, 1)
        self.checkBox_SavePSFVolume = QtGui.QCheckBox(self.layoutWidget)
        self.checkBox_SavePSFVolume.setObjectName(_fromUtf8("checkBox_SavePSFVolume"))
        self.gridLayout.addWidget(self.checkBox_SavePSFVolume, 12, 2, 1, 1)
        self.pushButton_StartCalc = QtGui.QPushButton(self.layoutWidget)
        self.pushButton_StartCalc.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_StartCalc.setFont(font)
        self.pushButton_StartCalc.setObjectName(_fromUtf8("pushButton_StartCalc"))
        self.gridLayout.addWidget(self.pushButton_StartCalc, 12, 4, 1, 2)
        self.line = QtGui.QFrame(self.layoutWidget)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.gridLayout.addWidget(self.line, 10, 0, 1, 6)
        self.label_Immersion = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_Immersion.setFont(font)
        self.label_Immersion.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_Immersion.setObjectName(_fromUtf8("label_Immersion"))
        self.gridLayout.addWidget(self.label_Immersion, 3, 3, 1, 1)
        self.immersion_text = QtGui.QLabel(self.layoutWidget)
        self.immersion_text.setAlignment(QtCore.Qt.AlignCenter)
        self.immersion_text.setObjectName(_fromUtf8("immersion_text"))
        self.gridLayout.addWidget(self.immersion_text, 3, 4, 1, 1)
        self.label_RI = QtGui.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_RI.setFont(font)
        self.label_RI.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_RI.setObjectName(_fromUtf8("label_RI"))
        self.gridLayout.addWidget(self.label_RI, 4, 3, 1, 1)
        self.ri_text = QtGui.QLabel(self.layoutWidget)
        self.ri_text.setAlignment(QtCore.Qt.AlignCenter)
        self.ri_text.setObjectName(_fromUtf8("ri_text"))
        self.gridLayout.addWidget(self.ri_text, 4, 4, 1, 1)

        self.retranslateUi(PSF_XYZ_Dialog_BF)
        QtCore.QMetaObject.connectSlotsByName(PSF_XYZ_Dialog_BF)

    def retranslateUi(self, PSF_XYZ_Dialog_BF):
        PSF_XYZ_Dialog_BF.setWindowTitle(_translate("PSF_XYZ_Dialog_BF", "PSF-XYZ Automatic Detection BF", None))
        self.checkBox_SavePeaks.setText(_translate("PSF_XYZ_Dialog_BF", "Save PSF Peaks FWHM", None))
        self.label_channel.setText(_translate("PSF_XYZ_Dialog_BF", "Select Channel", None))
        self.label_objNA.setText(_translate("PSF_XYZ_Dialog_BF", "NA", None))
        self.ExEm_text.setText(_translate("PSF_XYZ_Dialog_BF", "n.a.", None))
        self.text_filename.setText(_translate("PSF_XYZ_Dialog_BF", "imagefile", None))
        self.OpenFile.setText(_translate("PSF_XYZ_Dialog_BF", "Browse...", None))
        self.label_objname.setText(_translate("PSF_XYZ_Dialog_BF", "Objective", None))
        self.label_threshold.setToolTip(_translate("PSF_XYZ_Dialog_BF", "Threshold used to identify the peaks.", None))
        self.label_threshold.setText(_translate("PSF_XYZ_Dialog_BF", "Threshold [0-1]", None))
        self.label_pixsize.setToolTip(_translate("PSF_XYZ_Dialog_BF", "Pixel Size in image plane.", None))
        self.label_pixsize.setText(_translate("PSF_XYZ_Dialog_BF", "Pixel Size [nm]", None))
        self.zspacing.setToolTip(_translate("PSF_XYZ_Dialog_BF", "Distance between 2 focal planes.", None))
        self.zspacing.setText(_translate("PSF_XYZ_Dialog_BF", "Z-Spacing [micron]", None))
        self.label_guess_fwhmz.setToolTip(_translate("PSF_XYZ_Dialog_BF", "Guess for FWHM-Z used tfor the Gauss Fit.", None))
        self.label_guess_fwhmz.setText(_translate("PSF_XYZ_Dialog_BF", "Guess FWHM-Z [micron]", None))
        self.writexls.setText(_translate("PSF_XYZ_Dialog_BF", "Write Data to Excel File", None))
        self.label_ExEm.setText(_translate("PSF_XYZ_Dialog_BF", "EX/EM Wavelength [nm]", None))
        self.objname_text.setText(_translate("PSF_XYZ_Dialog_BF", "n.a.", None))
        self.label_subimage_size.setToolTip(_translate("PSF_XYZ_Dialog_BF", "Size of subimage which is cut out to apply the 2D Gauss Fit.", None))
        self.label_subimage_size.setText(_translate("PSF_XYZ_Dialog_BF", "Subimage Size [pixel]", None))
        self.label_offset.setToolTip(_translate("PSF_XYZ_Dialog_BF", "Size of subimage which is cut out to apply the 2D Gauss Fit.", None))
        self.label_offset.setText(_translate("PSF_XYZ_Dialog_BF", "Subtract Offset CCD [cts]", None))
        self.checkBox_SaveOrthoView.setText(_translate("PSF_XYZ_Dialog_BF", "Save PSF OrthoView", None))
        self.checkBox_SavePSFVolume.setText(_translate("PSF_XYZ_Dialog_BF", "Save PSF Volume", None))
        self.pushButton_StartCalc.setText(_translate("PSF_XYZ_Dialog_BF", "Start Detection", None))
        self.label_Immersion.setText(_translate("PSF_XYZ_Dialog_BF", "Immersion", None))
        self.immersion_text.setText(_translate("PSF_XYZ_Dialog_BF", "n.a.", None))
        self.label_RI.setText(_translate("PSF_XYZ_Dialog_BF", "Ref. Index", None))
        self.ri_text.setText(_translate("PSF_XYZ_Dialog_BF", "n.a.", None))

