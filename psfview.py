# -*- coding: utf-8 -*-
"""
@author: Sebi

psfview.py
Version: 0.4
Date: 2015-11-02
"""

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import visvis as vv
import skimage.feature as sf


def find_stackmax(imagestack):

    # find the zplane which contains the overall maximum position within the Z-stack
    overall_max = imagestack.max()
    position = (imagestack == overall_max).nonzero()
    zpos = position[0][0]
    # extract plane containing the brightest pixel
    planexy = imagestack[zpos, :, :]
    
    return zpos, planexy


def psf_orthoview(stack, width, z, ratio, filepath, threshold):

    # find brightest xy-plane and extract plane
    [zpos, planexy] = find_stackmax(stack)
    # peak detection with scikit-image - only 1 peak is allowed
    peaks = sf.peak_local_max(planexy, min_distance=1, threshold_rel=threshold,
                              exclude_border=True, indices=True, num_peaks=1)

    peaknum = len(peaks)
    xpos = np.zeros(len(peaks))
    ypos = np.zeros(len(peaks))
    for p in arange(len(peaks)):
        # x and y coordinates from skimage.peak_local_max are switched
        xpos[p] = peaks[p][1]
        ypos[p] = peaks[p][0]

    planexz = stack[:, ypos[0], :]
    planeyz = stack[:, :, xpos[0]]
    planeyz = np.rot90(planeyz)
    
    a1 = 1.0
    a2 = 1/ratio
    a3 = ratio

    # display image and detected peak
    fig = plt.figure(figsize=(7, 7))
    fig.canvas.set_window_title('Average PSF - OrthoView')
    ax1 = fig.add_subplot(2, 2, 1)  # xy
    ax2 = fig.add_subplot(2, 2, 2)  # xz
    ax3 = fig.add_subplot(2, 2, 3)  # yz
    ax1.imshow(planexy, interpolation='nearest', aspect=a1, extent=None, origin=None,  cmap=cm.jet)
    ax2.imshow(planeyz, interpolation='nearest', aspect=a2, extent=None, origin=None,  cmap=cm.jet)
    ax3.imshow(planexz, interpolation='nearest', aspect=a3, extent=None, origin=None,  cmap=cm.jet)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax2.set_xlabel('Z-Dimension')
    ax3.set_ylabel('Z-Dimension')
    ax1.set_title('XY-Plane')
    ax2.set_title('XZ-Plane')
    ax3.set_title('YZ-Plane')
    
    fig.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.95,wspace=0.1, hspace=0.2)

    # save screenshot
    if filepath != 'nosave':
        print 'Saving PSF OrthoView.'
        savename = filepath[:-4] + '_PSF_OrthoView.png'
        fig.savefig(savename)


def psf_volume(stack, xyz_ratio, filepath):

    app = vv.use()
    # Init a figure with two axes
    a1 = vv.subplot(121)
    vv.title('PSF Volume')
    a2 = vv.subplot(122)
    vv.title('PSF XYZ Cross Sections')
    
    # show
    t1 = vv.volshow(stack, axes=a1)  # volume
    t2 = vv.volshow2(stack, axes=a2)  # cross-section interactive
       
    # set labels for both axes
    vv.xlabel('Pixel X', axes=a1)
    vv.ylabel('Pixel Y', axes=a1)
    vv.zlabel('Z-Slice', axes=a1)
    vv.xlabel('Pixel X', axes=a2)
    vv.ylabel('Pixel Y', axes=a2)
    vv.zlabel('Z-Slice', axes=a2)
    
    # set colormaps
    t1.colormap = vv.CM_JET
    t2.colormap = vv.CM_JET

    # set correct aspect ration corresponding to voxel size    
    a1.daspect = 1, 1, xyz_ratio
    a2.daspect = 1, 1, xyz_ratio
    
    # show grid
    a1.axis.showGrid = 1
    a2.axis.showGrid = 1    
    
    # run visvis and show results
    app.Run()

    # save screenshot
    if filepath != 'nosave':
        print 'Saving PSF volume.'
        savename = filepath[:-4] + '_PSF_3D.png'
        # sf: scale factor
        vv.screenshot(savename, sf=1, bg='w')
