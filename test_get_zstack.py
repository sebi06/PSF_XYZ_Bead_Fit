import bfimage as bf
from matplotlib import pyplot as plt, cm
import os
import numpy as np

def adjust_max(stack, maxlimit):
    
    print 'Old Stackmax: ', stack.max()    
    
    if stack.max() > maxlimit:
        
        for z in range(0, stack.shape[0]):
            plane = stack[z, :, :]
            pos_max = (plane > maxlimit).nonzero()
            plane[pos_max] = np.round(plane.mean())
            stack[z, :, :] = plane
            print 'New PlaneMax : ', stack[z, :, :].max()
        
    print 'New Stackmax: ', stack.max()
            
    return stack


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

# get plane with the brightest pixel
zplane = (zstack == zstack.max()).nonzero()[0][0]

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

print 'New Max: ', stack.max()


#img2show = zstack[zplane, channel, :, :]
#fig1 = plt.figure(figsize=(10, 8), dpi=100)
#ax1 = fig1.add_subplot(111)
#cax = ax1.imshow(img2show, interpolation='nearest', cmap=cm.hot)
#ax1.set_title('T=' + str(timepoint+1) + ' Z=' + str(zplane+1) + ' CH=' + str(channel+1), fontsize=12)
#ax1.set_xlabel('X-dimension [pixel]', fontsize=10)
#ax1.set_ylabel('Y-dimension [pixel]', fontsize=10)
#cbar = fig1.colorbar(cax)
#ax1.format_coord = bf.Formatter(cax)
## show plots
#plt.show()

