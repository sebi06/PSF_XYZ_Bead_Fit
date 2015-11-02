# -*- coding: utf-8 -*-
"""
@author: Sebi

PSF_XYZ_Dialog_BF.py
Version: 1.5
Date: 2015-11-02

This program can be used to detect beads an measures the FWHM-XYZ of the PSF.
The crucial steps are:
    1) read the z-stack via BioFormats
    2) find the brightest voxel and uses the corresponding XY-plane !!! hot pixels may ruin the results !!!
    3) detect all peaks within the XY-plane from 2)
    4) extract the Z-profile at every detected peak position
    5) determine the brightest XY-plane for every peak separately
    6) cutout sub-images at every detected peak position
    7) do 2D-Gauss fit for every peak to determine FWHM-XY
    8) do 1D-Gauss fit for every Z-Profile to determine FWHM-Z
    9) displays PSF OrthoView and PSF volume for the average PSF = sum of all detected PSFs
    10) optional - displays PSF OrthoView and PSF volume for one randomly selected peak (optional)
    11) optional - write results to excel sheet (currently only XLS)
    12) optional - save output graphics as PNGs
"""

version = 1.5

from pylab import *
import numpy as np
import gaussfit as gf
import psfview as psf
import matplotlib.pyplot as plt
from xlwt import Workbook
import os
from remove_hotpix import adjust_max
import bfimage as bf
import skimage.feature as sf


from PyQt4.QtGui import *
# Qt4 bindings for core Qt functionality (non-GUI)
from PyQt4 import QtCore
# Python Qt4 bindings for GUI objects
from PyQt4 import QtGui

# import the MainWindow widget from the converted .ui files
import ui_PSF_XYZ_Dialog_BF


class PSF_XYZ_Dialog_BF(QDialog, ui_PSF_XYZ_Dialog_BF.Ui_PSF_XYZ_Dialog_BF):

    def __init__(self, parent=None):
        super(PSF_XYZ_Dialog_BF, self).__init__(parent)
        self.setupUi(self)

        # set window title to current version
        self.setWindowTitle('PSF-XYZ Automatic Detection BF ' + str(version))

        # connect the signals with the slots

        # Browse Image File
        QtCore.QObject.connect(self.OpenFile, QtCore.SIGNAL('clicked()'), self.onopen_file)
        # Start PSF Detection
        QtCore.QObject.connect(self.pushButton_StartCalc, QtCore.SIGNAL('clicked()'), self.onstart_detection)
        # Select Channel
        QtCore.QObject.connect(self.SpinBox_channel, QtCore.SIGNAL('valueChanged(int)'), self.onchannel_changed)
        # Activate HotPixel Removal Option
        QtCore.QObject.connect(self.check_hotpixel, QtCore.SIGNAL('stateChanged(int)'), self.onremove_hotpix_changed)

        # initialize dictionaries
        self.MetaInfo = {}
        self.BeadData = {}

    def onopen_file(self):
        """
        open image file dialog with default starting directory
        """
        #default_folder = os.getcwd()
        default_folder = r'f:\Castor_Image_Data\Castor_Beta\Castor_Beta2\20150923'

        psfstack_filepath = QtGui.QFileDialog.getOpenFileName(self, 'Open file',
                                default_folder, 'CZI Files (*.czi);; TIF Files (*.tif);; TIFF Files (*.tiff)')

        # update filename inside the GUI
        self.text_filename.setText(psfstack_filepath)

        # get image data file location
        imagefilepath = str(self.text_filename.text())

        # specify bioformats_package.jar to use if required
        #bf.set_bfpath('c:\Users\M1SRH\Documents\Software\BioFormats_Package\5.1.4\bioformats_package.jar')
        #bf.set_bfpath('c:\Users\M1SRH\Documents\Spyder_Projects\PSF_XYZ_Bead_Fit_BioFormats\bioformats_package.jar')

        # get the relevant MetaData
        self.MetaInfo = bf.bftools.get_relevant_metainfo_wrapper(imagefilepath)

        print 'Image Directory      : ', self.MetaInfo['Directory']
        print 'Image Filename       : ', self.MetaInfo['Filename']
        print 'Images Dim Sizes     : ', self.MetaInfo['Sizes']
        print 'Dimension Order BF   : ', self.MetaInfo['DimOrder BF']
        print 'Dimension Order CZI  : ', self.MetaInfo['OrderCZI']
        print 'Total Series Number  : ', self.MetaInfo['TotalSeries']
        print 'Image Dimensions     : ', self.MetaInfo['TotalSeries'], self.MetaInfo['SizeT'], self.MetaInfo['SizeZ'],\
            self.MetaInfo['SizeC'], self.MetaInfo['SizeY'], self.MetaInfo['SizeX']
        print 'Scaling XYZ [micron] : ', self.MetaInfo['XScale'], self.MetaInfo['YScale'], self.MetaInfo['ZScale']
        print 'Objective M-NA-Imm   : ', self.MetaInfo['ObjMag'], self.MetaInfo['NA'], self.MetaInfo['Immersion']
        print 'Objective Name       : ', self.MetaInfo['ObjModel']
        print 'Detector Name        : ', self.MetaInfo['DetName']
        print 'Ex. Wavelengths [nm] : ', self.MetaInfo['WLEx']
        print 'Em. Wavelengths [nm] : ', self.MetaInfo['WLEm']
        print 'Channel Description  : ', self.MetaInfo['ChDesc']

        self.objname_text.setText(self.MetaInfo['ObjModel'])
        if self.MetaInfo['NA'] != 'n.a.':
            self.SpinBox_NA.setValue(self.MetaInfo['NA'])
        self.onchannel_changed()
        self.SpinBox_pixsize.setValue(self.MetaInfo['XScale']*1000)
        self.SpinBox_zspacing.setValue(self.MetaInfo['ZScale'])

        # enable button to actually start the PSF detection process
        self.pushButton_StartCalc.setEnabled(True)

        # update estimated FWHM-Z
        estimate, refindex_n = estimate_fwhmz(self.MetaInfo['NA'], self.MetaInfo['WLEm'][0], self.MetaInfo['Immersion'])
        self.SpinBox_guess_fwhmz.setValue(estimate)
        self.immersion_text.setText(self.MetaInfo['Immersion'])
        self.ri_text.setText(str(refindex_n))

        # limit possible values for channel based on MetaInfo
        self.SpinBox_channel.setMaximum(self.MetaInfo['SizeC'])
        # disable channel selection if there is only one channel
        if self.MetaInfo['SizeC'] < 2:
            self.SpinBox_channel.setEnabled(False)

    def find_peaks(self, planexy):
        """
        Find the peaks inside the plane with the brightest pixel using skimage library.
        :param planexy: 2D image
        :return: xpos, ypos, peaknum
        """

        # get minimal distance --> use subimage size / 2
        mindist = np.round(self.SpinBox_subimage_size.value(), 0) + 1
        th = self.SpinBox_threshold.value()
        # peak detection with scikit-image
        peaks = sf.peak_local_max(planexy, min_distance=mindist, threshold_rel=th, exclude_border=True,
                                  indices=True, num_peaks=inf)

        peaknum = len(peaks)
        xpos = np.zeros(len(peaks))
        ypos = np.zeros(len(peaks))
        for p in arange(len(peaks)):
            # x and y coordinates from skimage.peak_local_max are switched
            xpos[p] = peaks[p][1]
            ypos[p] = peaks[p][0]

        # print('Detected Peak Positions : ', xpos, ypos)
        print 'Number of Peaks : ', peaknum

        # return plist, xpos, ypos, peaknum
        return xpos, ypos, peaknum

    def fit_psf(self, peaknum, xdim, ydim, zdim, stack, xpos, ypos):

        # create matrix for Z-profiles
        zprofiles = np.zeros([zdim, peaknum])

        # create vector containing the maximum value of every Z-profile
        zprofiles_max = np.zeros(peaknum)

        # extract z-profiles at every detected peak
        for i in range(0, peaknum, 1):
            zprofiles[:, i] = stack[:, ypos[i], xpos[i]]  # swap coordinates !!!
            # write highest value in vector
            zprofiles_max[i] = zprofiles[:, i].max()

        # create vector for the position of the maximum within every Z-profile
        zprofiles_max_pos = np.zeros(peaknum)

        # determine position of brightest pixel for every z-profile
        for i in range(0, peaknum, 1):
            maxposition = (zprofiles[:, i] == zprofiles_max[i]).nonzero()
            # in case there are more maxima along the z-axis just take the mean ...
            zprofiles_max_pos[i] = np.int(np.mean(maxposition[0]))

        print 'Z-Profiles MAX Values:', zprofiles_max
        print 'Z-Profiles MAX Positions:', zprofiles_max_pos

        # loop through all peak positions and return imagelist (igl)
        igl = cut_subimages(peaknum, xpos, ypos, zprofiles_max_pos, self.SpinBox_subimage_size.value(), stack)

        # initialize data matrix
        results = zeros((peaknum, 9))
        xfit_z = np.zeros([zdim, peaknum])
        yfit_z = np.zeros([zdim, peaknum])

        # do the PSF-XY fits using the plane which corresponds to the Z-position
        # of the maximum value extracted from the Z-profiles
        for i in range(0, peaknum, 1):

            # fit PSF-XY in xy-plane using 2D-Gauss
            params = gf.fitgaussian2D(igl[i])
            fit = gf.gaussian2D(*params)
            (height, bgrd, x, y, width_x, width_y) = params
            fwhm_x = width_x * self.SpinBox_pixsize.value() * 2.3548
            fwhm_y = width_y * self.SpinBox_pixsize.value() * 2.3548
            results[i, 0] = round(x, 3)        # center x
            results[i, 1] = round(y, 3)        # center y
            results[i, 2] = round(height, 0)   # height of peak
            results[i, 3] = round(bgrd, 0)     # background
            results[i, 4] = round(fwhm_x, 0)   # FWHM-X
            results[i, 5] = round(fwhm_y, 0)   # FWHM-Y

            # vector for spacing in dz
            zrange = arange(0, stack.shape[0], 1) * self.SpinBox_zspacing.value()

            # guess z peak position from profile
            z_peak_positions = zprofiles_max_pos * self.SpinBox_zspacing.value()

            # fit PSF-Z using the Z-profiles with 1D-Gauss
            [bgrd, heightZ, center, fwhm_z, cov, xfit_z[:, i], yfit_z[:, i]] = gf.fitgaussian1D(zrange, zprofiles[:, i],
                                    self.SpinBox_guess_fwhmz.value(), z_peak_positions[i])

            results[i, 6] = round(fwhm_z*1000, 0)  # FWHM-Z
            results[i, 7] = zprofiles_max_pos[i]   # Z-Planes
            results[i, 8] = zprofiles_max[i]       # Brightest Pixel

        heightXY = results[:, 2]
        bgrdXY = results[:, 3]
        fwhmx = np.abs(results[:, 4])
        fwhmy = np.abs(results[:, 5])
        fwhmz = results[:, 6]
        zplanes_pos_all = results[:, 7]
        zplanes_max_all = results[:, 8]
        fwhmxy_all = np.concatenate((fwhmx, fwhmy))
        fwhmxy_all_ok = (fwhmxy_all > 100).nonzero()

        print ('FWHM-X [nm] : '), fwhmx
        print ('FWHM-Y [nm] : '), fwhmy
        print ('FWHM-Z [nm] : '), fwhmz

        return heightXY, bgrdXY, fwhmx, fwhmy, fwhmz, zplanes_pos_all, zplanes_max_all, fwhmxy_all, fwhmxy_all_ok, igl, fit

    def display_results(self, xdim, ydim, zdim, stack, imagefilepath, planexy, xpos, ypos, zpos,
                       fwhmx, fwhmy, fwhmz, heightXY, bgrdXY, fwhm_all, fwhm_all_ok, igl, fit):

        # Gauss 2D fit for randomly selected peak
        goodimages = (fwhmx > 0).nonzero()
        tmp = goodimages[0]
        rn = int(round(random(1)*(len(tmp)-1), 0))
        img2show = int(tmp[rn])

        # display image and detected peaks
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.set_window_title(imagefilepath)
        ax1 = fig.add_subplot(2, 2, 1)  # all detected peaks
        ax2 = fig.add_subplot(2, 2, 2)  # random selected peak with fit
        ax3 = fig.add_subplot(2, 2, 3)  # FWHM-XY distribution
        ax4 = fig.add_subplot(2, 2, 4)  # FWHM-Z example profile
        fig.subplots_adjust(left=0.07, bottom=0.1, right=0.97, top=0.92, wspace=0.20, hspace=0.25)

        # ax1
        ax1.imshow(planexy, interpolation='nearest', origin='None',  cmap=cm.jet)
        ax1.plot(xpos, ypos, 'rs', markersize=12, markeredgewidth=1, alpha=0.3)
        ax1.plot(xpos[img2show], ypos[img2show], 'ys', markersize=20, markeredgewidth=1, alpha=0.4)
        ax1.axis([0, xdim, 0, ydim])
        ax1.set_xlabel('pixel x')
        ax1.set_ylabel('pixel y')
        ax1.set_title('Peak Detection and FWHM-XY Fit', fontsize=12)

        # ax2
        cax2 = ax2.imshow(igl[img2show], interpolation='nearest', origin=None,  cmap=cm.jet)
        ax2.set_xlabel('pixel x')
        ax2.set_ylabel('pixel y')
        ax2.set_title('Random Peak Image shown : ' + str(img2show), fontsize=12)
        ax2.contour(fit(*indices(igl[img2show].shape)), cmap=cm.copper)
        ax2.text(0.90, 0.90, """
        Channel : %.0f""" %(self.SpinBox_channel.value()),
            fontsize=12, fontweight='bold', horizontalalignment='right', color='red',
            verticalalignment='bottom', transform=ax2.transAxes)

        ax2.text(0.95, 0.05, """
        Height  : %.0f
        Bgrd    : %.0f
        FWHM-X  : %.0f
        FWHM-Y  : %.0f
        FWHM-Z  : %.0f""" %(np.round(heightXY[img2show], 0), round(bgrdXY[img2show], 0),
            fwhmx[img2show], fwhmy[img2show], fwhmz[img2show]),
            fontsize=12, horizontalalignment='right', color='red',
            verticalalignment='bottom', transform=ax2.transAxes)

        # ax3
        # the histogram of the data
        n, bins, patches = ax3.hist(fwhm_all[fwhm_all_ok], 10, label='FWHM-XY',
                                    normed=0, facecolor='green', alpha=0.75)
        ax3.set_xlabel('FWHM-XY [nm]')
        ax3.set_ylabel('Occurrence')
        ax3.set_title('Measured FWHM-XY', fontsize=12)
        ax3.set_xlim(fwhm_all[fwhm_all_ok].min()*0.95, fwhm_all[fwhm_all_ok].max() * 1.05)
        ax3.legend()
        ax3.grid(True)

        # ax4
        fwhmz_ok = fwhmz.ravel().nonzero()

        try:
            n, bins, patches = ax4.hist(fwhmz[fwhmz_ok], 10, label='FWHM-Z', normed=0, facecolor='green', alpha=0.75)
        except:
            print "Only one data point --> no histogram plotted."
            ax4.plot([fwhmz[fwhmz_ok], fwhmz[fwhmz_ok]], [0, 1], 'g-', lw=5, label='FWHM-Z')

        ax4.set_xlabel('FWHM-Z [nm]')
        ax4.set_ylabel('Occurrence')
        ax4.set_title('Measured FWHM-Z', fontsize=12)
        ax4.set_xlim(fwhmz[fwhmz_ok].min()*0.95, fwhmz[fwhmz_ok].max() * 1.05)
        ax4.legend()
        ax4.grid(True)

        # only save plot when option is checked
        if self.checkBox_SavePeaks.isChecked() == True:

            print 'Saving PSF peaks.'
            savename =  self.BeadData['FileDir']+'/'+self.BeadData['FileName'][:-4] + '_PSF_FWHM.png'
            fig.savefig(savename)

        # display PSF-OrthoView for selected peak
        # estimate a "good" number of pixels around center of PSF to be displayed
        psfwidth = np.round(np.mean(fwhm_all) / self.MetaInfo['XScale']/1000, 0) * 3
        # estimate a "good" number of planes below and above PSF center
        dz = np.round((np.mean(fwhmz) / self.MetaInfo['ZScale']/1000), 0) * 3

        # dz dimension check --> too small
        if dz*2 > self.MetaInfo['SizeZ']:
            dz = np.round(self.MetaInfo['SizeZ']/2)

        ratio = self.SpinBox_zspacing.value() / self.SpinBox_pixsize.value()*1000  # pixel in [nm] !!!
        print 'Aspect Ratio Z-XY : ', ratio

        PSFstack_avg = calc_average_psf(stack, xpos, ypos, zdim, psfwidth)

        zstart, zend = limit_zrange(zdim, dz)

        # only save plot when option is checked
        if self.checkBox_SaveOrthoView.isChecked() == True:
            psf.psf_orthoview(PSFstack_avg[zstart:zend, :, :], psfwidth, dz, ratio, self.BeadData['FilePath'],
                              threshold=self.SpinBox_threshold.value())
        elif self.checkBox_SavePeaks.isChecked() == False:
            psf.psf_orthoview(PSFstack_avg[zstart:zend, :, :], psfwidth, dz, ratio, 'nosave',
                              threshold=self.SpinBox_threshold.value())

        # only save plot when option is checked
        if self.checkBox_SavePSFVolume.isChecked() == True:
            psf.psf_volume(PSFstack_avg[zstart:zend, :, :], ratio, self.BeadData['FilePath'])
        elif self.checkBox_SavePSFVolume.isChecked() == False:
            psf.psf_volume(PSFstack_avg[zstart:zend, :, :], ratio, 'nosave')

    def onchannel_changed(self):

        newch = self.SpinBox_channel.value()
        # get EX/EM wavelength for the selected channel
        ex = self.MetaInfo['WLEx'][newch - 1]
        em = self.MetaInfo['WLEm'][newch - 1]
        exem = str(ex) + ' / ' + str(em) + 'nm'
        print 'New Ex-Em: ', exem
        self.ExEm_text.setText(exem)

    def onremove_hotpix_changed(self):

        if self.check_hotpixel.isChecked() == True:

            self.SpinBox_hotpix_thresh.setEnabled(True)
            self.SpinBox_tolerance.setEnabled(True)
            self.SpinBox_kernelsize.setEnabled(True)

        elif self.check_hotpixel.isChecked() == False:

            self.SpinBox_hotpix_thresh.setEnabled(False)
            self.SpinBox_tolerance.setEnabled(False)
            self.SpinBox_kernelsize.setEnabled(False)

    def onstart_detection(self):

        """A Python dictionary will be created to hold the relevant Metadata."""

        self.BeadData = {'Height': 0,
                         'Background': 0,
                         'FWHM-X': 0,
                         'FWHM-Y': 0,
                         'FWHM-Z': 0,
                         'Z-Plane': 0,
                         'BrightestPixel': 0,
                         'FWHM-XY-All': 0,
                         'FWHM-XY-All-OK': 0,
                         'FilePath': 'na',
                         'FileName': 'na',
                         'FileDir': 'na',
                         'X-Dim': 0,
                         'Y-Dim': 0,
                         'Z-Dim': 0}

        # read image data location
        imagefilepath = str(self.text_filename.text())
        imagefile = os.path.basename(imagefilepath)
        imagedir = os.path.dirname(imagefilepath)

        # fill BeadData
        self.BeadData['FilePath'] = imagefilepath
        self.BeadData['FileName'] = imagefile
        self.BeadData['FileDir'] = imagedir

        seriesID = 0   # set series ID to 0
        timepoint = 0  # set timepoint to 0

        # get the z-stack using BioFormats
        channel = self.SpinBox_channel.value() - 1
        imagestack = np.squeeze(bf.bftools.get_zstack(imagefilepath, self.MetaInfo['Sizes'], seriesID, timepoint)[:, channel, :, :])

        # option removal of potential hot pixels
        if self.check_hotpixel.isChecked() == True:

            th = self.SpinBox_hotpix_thresh.value()
            tl = self.SpinBox_tolerance.value()
            ks = self.SpinBox_kernelsize.value()

            if imagestack.max() > th:
                # remove a stripe with a certain width from all edges
                re = 3
                # process the z-Stack and remove potential hotpixel
                imagestack = adjust_max(imagestack, maxlimit=th, remove_edge=re, kernelsize=ks, tolerance=tl)
        
        # offset subtraction if image contains an offset
        imagestack = imagestack - self.SpinBox_offset.value()

        # find brightest xy-plane and extract this plane
        zpos, planexy = find_stackmax(imagestack)

        # check dimensions --> switched !!!
        xdim = imagestack.shape[2]  # x-dimension
        ydim = imagestack.shape[1]  # y-dimension
        zdim = imagestack.shape[0]  # z-dimension

        # fill BeadData
        self.BeadData['X-Dim'] = imagestack.shape[2]
        self.BeadData['Y-Dim'] = imagestack.shape[1]
        self.BeadData['Z-Dim'] = imagestack.shape[0]

        # for testing
        #print xdim, ydim, zdim

        # PeakFind algorithm for pre-detection of beads
        xpos, ypos, peaknum = self.find_peaks(planexy)

        # Fit the PSF for all detected peaks using Gauss fit
        [heightXY, bgrdXY, fwhmx, fwhmy, fwhmz, zplanes_pos_all, zplanes_max_all, fwhmxy_all, fwhmxy_all_ok, igl, fit] =\
            self.fit_psf(peaknum, xdim, ydim, zdim, imagestack, xpos, ypos)

        self.BeadData['Height'] = heightXY
        self.BeadData['Background'] = bgrdXY
        self.BeadData['FWHM-X'] = fwhmx
        self.BeadData['FWHM-Y'] = fwhmy
        self.BeadData['FWHM-Z'] = fwhmz
        self.BeadData['Z-Plane'] = zplanes_pos_all
        self.BeadData['BrightestPixel'] = zplanes_max_all
        self.BeadData['FWHM-XY-All-OK'] = fwhmxy_all_ok

        # create XLS sheet when option is checked
        if self.writexls.isChecked() == True:

            create_xls(heightXY, bgrdXY, fwhmx, fwhmy, fwhmz, zplanes_pos_all, zplanes_max_all, fwhmxy_all,
                           fwhmxy_all_ok, self.SpinBox_channel.value(), imagedir, imagefile)

        # display the results
        self.display_results(xdim, ydim, zdim, imagestack, imagefilepath, planexy, xpos, ypos, zpos, fwhmx, fwhmy,
                            fwhmz, heightXY, bgrdXY, fwhmxy_all, fwhmxy_all_ok, igl, fit)

        plt.show()

        # reset the button
        self.pushButton_StartCalc.setEnabled(False)


def cut_subimages(peaknumber, peakx, peaky, zplanes, subimagesize, stack):

    # this functions cuts out subimages at detected peak positions

    sz = np.round(subimagesize/2, 0) + 1
    # initialize empty list for the subimages
    imagelist = [None]*peaknumber

    for i in range(0, peaknumber, 1):
        # xy must be switched due numpy´s array indexing
        # cutout subimage at every peak position at the corresponding z-position
        imgdata = stack[zplanes[i]]
        subimage = imgdata[peaky[i]-sz:peaky[i]+sz+1, peakx[i]-sz:peakx[i]+sz+1]
        # store subimage inside a list
        imagelist[i] = subimage
        #print 'XY Subimage : ', subimage.shape[0], subimage.shape[1]

    return imagelist


def find_stackmax(stack):
    # finds the zplane with contains the overall maximum position within the stack

    # get the brightest voxel
    overall_max = stack.max()
    position = (stack == overall_max).nonzero()
    zpos = position[0][0]

    # extract plane containing the brightest pixel
    planexy = stack[zpos]

    return zpos, planexy


def calc_average_psf(imagestack, xpos, ypos, zdim, psfwidth):

    # initialize stack to hold the average PSF
    avgPSF = np.zeros([zdim, 2 * psfwidth, 2 * psfwidth])

    # sum up all detected PSFs
    bad_psf = 0
    for i in range(0, len(xpos)):
        try:
            avgPSF = avgPSF + imagestack[:, ypos[i]-psfwidth:ypos[i] + psfwidth, xpos[i] - psfwidth:xpos[i] + psfwidth]
        except:
            # it might happen that the avgPSF stack was initialized with a size so thatit tries to include peaks
            # that to close to the border of the image. Those are exclude from the averaging only!
            print 'PSF Volume for Peak: ', ypos[i], xpos[i], ' was excluded from averaging.'
            bad_psf += 1

    # calc the mean of all detected PSFs minus the ones that are too close for the average stack
    avgPSF = np.round(avgPSF/(len(xpos)-bad_psf), 0)

    return avgPSF


def limit_zrange(zdim, dz):

    # limit range of PSF stack to be displayed and check values
    zstart = np.round(zdim/2) - dz
    zend = np.round(zdim/2) + dz
    if zstart < 0:
        zstart = 0
    if zend > zdim:
        zend = zdim

    return zstart, zend


def create_xls(heightXY, bgrdXY, fwhmx, fwhmy, fwhmz, zplanes_pos_all, zplanes_max_all, fwhm_all, fwhm_all_ok,
               channel, imagedir, imagefile):

    #TODO: Implement writing XLSX File

    # create numbers for peaklist
    peaklist = np.arange(0, fwhmx.shape[0], 1)

    paramlist = ['Peak Number',
                 'FWHM-X [nm]',
                 'FWHM-Y [nm]',
                 'FWHM-Z [nm]',
                 'Z-Plane',
                 'Height [cts]',
                 'Bgrd [cts]']

    valuelist = ['peaklist[i]+1',
                 'fwhmx[i]',
                 'fwhmy[i]',
                 'fwhmz[i]',
                 'zplanes_pos_all[i]',
                 'heightXY[i]',
                 'bgrdXY[i]']

    book = Workbook()
    # ---------------------------------
    sheet1 = book.add_sheet('PSF FWHM-XYZ Values')
    # ------------------------------------
    sheet1.col(0).width = 4000
    sheet1.col(1).width = 4000
    sheet1.col(2).width = 4000
    sheet1.col(3).width = 4000

    sheet1.write(0, 0, 'Directory')
    sheet1.write(1, 0, 'File')
    sheet1.write(0, 1, imagedir)
    sheet1.write(1, 1, imagefile)
    sheet1.write(2, 0, 'Channel')
    sheet1.write(2, 1, channel)

    for i in range(0, len(paramlist)):
        sheet1.write(4, i, paramlist[i])

    for i in range(0, len(peaklist), 1):
        for j in range(0, len(valuelist), 1):
            value = np.double(eval(valuelist[j]))
            sheet1.write(i+5, j, value)

    # save XLS file
    print 'Save Excel to: ', imagedir+imagefile+'_CH='+str(channel)+'_FWHM_XYZ.xls'
    os.chdir(imagedir)
    book.save(imagefile+'_CH='+str(channel)+'_FWHM_XYZ.xls')


def estimate_fwhmz(na=1.0, wl=0.515, imm='Air'):
    # this is JUST an estimation how to come up with a good guess for the FWHM-Z value

    # additional check for valid values
    if na == 'n.a.':
        na = 1.0  # just assume air objective
    if wl == 0:
        wl = 525  # assume emission wavelength = 525nm

    if imm == 'Air' or imm == 'air' or imm == 'Luft' or imm == 'luft':
        n = 1.0  # Air Objective used
    elif imm == 'Water' or imm == 'water' or imm == 'Wasser' or imm == 'wasser':
        n = 1.33  # Water Immersion Objective used
    elif imm == 'Oil' or imm == 'oil' or imm == 'Öl' or imm == 'öl' or imm == 'Oel' or imm == 'oel':
        n = 1.518  # Oil Immersion Objective used
    elif imm == 'n.a.':
        if na > 1.33:
            n = 1.518  # Oil Immersion objective used
        elif 1.0 < na < 1.33:
            n = 1.33  # Water Immersion Objective used
        elif na < 1.0:
            n = 1.0  # Air objective used
        print "Assuming refractive index: ", n

    if wl == 0:
        wl = 0.515  # set to 515nm as default
        print "Assuming emission wavelength: ", wl

    if na == 0:
        na = 1.0  # set default NA to 1.0 if unknown
        print "Assuming objective NA: ", na

    fwhmz_estimate = np.sqrt(2) * n * (wl/1000) / (na**2)
    # the PSF values are normally never as good as the theory predicts ...
    out = np.round(fwhmz_estimate * 1.2, 1)

    return out, n

####################################################################################################################

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    form = PSF_XYZ_Dialog_BF()
    form.show()
    app.exec_()
