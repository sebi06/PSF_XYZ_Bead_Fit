# -*- coding: utf-8 -*-
"""
@author: sebi

remove_hotpix.py
Version: 0.2
Date: 2015-11-02
"""

import numpy as np
from scipy.ndimage import median_filter


def find_hotpixel(data, tolerance=20, kernelsize=2):

    # use median filer to blur the image
    blurred = median_filter(data, size=kernelsize)
    # calculate the difference image
    difference = np.abs(data - blurred)
    # calculate the threshold
    threshold = tolerance * np.std(difference)
    # find the hot pixels
    hot_pixels = np.nonzero((np.abs(difference) > threshold))
    hot_pixels = np.array(hot_pixels)
    # remove the found hotpixel and replace them with the value from the blurred image
    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    return hot_pixels, fixed_image


def adjust_max(stack, maxlimit=16384, remove_edge=3, tolerance=20, kernelsize=2):
    
    print('Old Stackmax: ', stack.max())
    print('MaxLimit ', maxlimit, ' Tolerance :', tolerance, ' KernelSize: ', kernelsize)
    
    # remove some pixel from the edges
    newstack = np.int32(stack[:, remove_edge:-remove_edge, remove_edge:-remove_edge])
    
    if newstack.max() > maxlimit:
        
        # cycle through the stack and check for potential hotpixel plane-by-plane
        for z in range(0, newstack.shape[0]):
            hotpix, plane = find_hotpixel(newstack[z, :, :], tolerance=tolerance, kernelsize=kernelsize)
            newstack[z, :, :] = plane
        
    print('New Stackmax: ', newstack.max())
            
    return newstack
