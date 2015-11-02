# -*- coding: utf-8 -*-
"""
@author: sebi

gaussfit.py
Version: 0.4
Date: 2015-11-02
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import optimize
from pylab import *


# define your fitting function
def gaussian1D(x, bgrd, height, center, sigma):

    """Gauss 1D function y = a + (b-a)*exp(-(x-c)**2/(2*d**2))"""

    return bgrd + (height - bgrd)*np.exp(-(x-center)**2/(2*sigma**2))


def fitgaussian1D(xdat, ydat, fwhm_guess, peak_pos):

    # guess initial parameters from the original data
    bgrd_guess = ydat.min()
    maxpeak = ydat.max()
    #mean_guess = sum(xdat*ydat)/sum(ydat)
    center_guess = peak_pos

    # define parameter set for initial guess
    #p0=[bgrd_guess,maxpeak,mean_guess,fwhm_guess/2.3548]
    p0 = [bgrd_guess, maxpeak, center_guess, fwhm_guess/2.3548]

    # do the least square fit using the Levenberg-Marquardt algorithm
    popt, pcov = curve_fit(gaussian1D, xdat, ydat, p0)

    bgrd = popt[0]
    height = popt[1]
    center = popt[2]
    fwhm = popt[3] * 2.3548

    # get std from the diagonal of the covariance matrix
    #std_bgrd = np.sqrt(pcov1[0,0])
    #std_height = np.sqrt(pcov1[1,1])
    #std_center = np.sqrt(pcov1[2,2])
    #std_sigma = np.sqrt(pcov1[3,3])

    #print ('Background : %.3f' % popt[0])
    #print ('Height     : %.3f' % popt[1])
    #print ('Center     : %.3f' % popt[2])
    #print ('FWHM       : %.3f' % (popt[3] * 2.3548))
    #print ('STD Bgrd   : %.3f' % std_bgrd)
    #print ('STD Height : %.3f' % std_height)
    #print ('STD Center : %.3f' % std_mean)
    #print ('STD FWHM   : %.3f' % (std_sigma * 2.3548))


    # calculate data for plot using the fitted parameters
    #yfit = fitfunc(xdat_new, popt[0], popt[1], popt[2],popt[3])
    yfit = gaussian1D(xdat, popt[0], popt[1], popt[2], popt[3])

    #return bgrd, height, center, fwhm, pcov, xdat_new, yfit
    return bgrd, height, center, fwhm, pcov, xdat, yfit


def gaussian2D(height, bgrd, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""

    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: bgrd + (height-bgrd)*exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


def moments(data):
    """
    Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments
    """

    total = data.sum()
    X, Y = indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    bgrd = data.min()

    return height, bgrd, x, y, width_x, width_y


def fitgaussian2D(data):
    """Returns (height, bgrd, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""

    params = moments(data)
    errorfunc = lambda p: ravel(gaussian2D(*p)(*indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunc, params)

    return p
