import bfimage as bf
from matplotlib import pyplot as plt, cm
import os
import numpy as np
from scipy.ndimage import median_filter

def adjust_max(stack, maxlimit, remove_edge=3):
    
    print 'Old Stackmax: ', stack.max()    
    
    # remove some pixel from the edges    
    newstack = np.int32(stack[:, remove_edge:-remove_edge, remove_edge:-remove_edge])
    
    if newstack.max() > maxlimit:
        
        for z in range(0, newstack.shape[0]):
            hotpix, plane = find_outlier_pixels(newstack[z, :, :])

            #pos_max = (plane > maxlimit).nonzero()
            #plane[pos_max] = maxlimit
            newstack[z, :, :] = plane
            #print 'New PlaneMax : ',stack[z,:,:].max()
        
    print 'New Stackmax: ', newstack.max()
            
    return newstack


def find_outlier_pixels(data, tolerance=20, kernelsize=2):
    # This function finds the hot or dead pixels in a 2D dataset.
    # tolerance is the number of standard deviations used to cutoff the hot pixels
    # If you want to ignore the edges and greatly speed up the code, then set
    # worry_about_edges to False.
    #
    # The function returns a list of hot pixels and also an image with with hot pixels removed

    blurred = median_filter(data, size=kernelsize)
    difference = np.abs(data - blurred)
    threshold = tolerance * np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference) > threshold))
    hot_pixels = np.array(hot_pixels) # because we ignored the first row and first column

    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]

    return hot_pixels,fixed_image

filename = r'c:\Users\M1SRH\Documents\Testdata_Zeiss\Castor\Beta_Bonn\20150923\PSF_20X_0.7_2X_488nm_dz=0.25_D153-173.czi'

imgbase = os.path.basename(filename)
imgdir = os.path.dirname(filename)

# specify bioformats_package.jar to use if required
#bf.set_bfpath(insert path to bioformats_packe.jar here)

## get image meta-information
MetaInfo = bf.bftools.get_relevant_metainfo_wrapper(filename)

seriesID = 0
timepoint = 0
channel = 0

# get the actual z-stack from the data set
zstack = np.squeeze(bf.bftools.get_zstack(filename, MetaInfo['Sizes'], seriesID, timepoint)[:, channel, :, :])


# show relevant image Meta-Information
print 'Image Directory      : ', imgdir
print 'Image Filename       : ', imgbase
print 'Images Dim Sizes     : ', MetaInfo['Sizes']
print 'Dimension Order BF   : ', MetaInfo['DimOrder BF']
print 'Dimension Order CZI  : ', MetaInfo['OrderCZI']
print 'Total Series Number  : ', MetaInfo['TotalSeries']
print 'Image Dimensions     : ', MetaInfo['TotalSeries'], MetaInfo['SizeT'], MetaInfo['SizeZ'], MetaInfo['SizeC'],\
                                    MetaInfo['SizeY'], MetaInfo['SizeX']
print 'Scaling XYZ [micron] : ', MetaInfo['XScale'], MetaInfo['YScale'], MetaInfo['ZScale']
print 'Objective M-NA-Imm   : ', MetaInfo['ObjMag'], MetaInfo['NA'], MetaInfo['Immersion']
print 'Objective Name       : ', MetaInfo['ObjModel']
print 'Detector Name        : ', MetaInfo['DetName']
print 'Ex. Wavelengths [nm] : ', MetaInfo['WLEx']
print 'Em. Wavelengths [nm] : ', MetaInfo['WLEm']
print 'Dyes                 : ', MetaInfo['Dyes']
print 'Channel Description  : ', MetaInfo['ChDesc']
print '============================================================='
print 'Shape Z-Stack        : ', np.shape(zstack)


stack = adjust_max(zstack, 6000)

# get plane with the brightest pixel
zplane = (stack == stack.max()).nonzero()[0][0]

img2show = stack[zplane, :, :]
fig1 = plt.figure(figsize=(10, 8), dpi=100)
ax1 = fig1.add_subplot(111)
cax = ax1.imshow(img2show, interpolation='nearest', cmap=cm.hot, vmin=img2show.min(), vmax=img2show.max())
ax1.set_title('T=' + str(timepoint+1) + ' Z=' + str(zplane+1) + ' CH=' + str(channel+1), fontsize=12)
ax1.set_xlabel('X-dimension [pixel]', fontsize=10)
ax1.set_ylabel('Y-dimension [pixel]', fontsize=10)
cbar = fig1.colorbar(cax)
ax1.format_coord = bf.Formatter(cax)
# show plots
plt.show()

