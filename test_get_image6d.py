import numpy as np
import os
import bfimage as bf

#filename = r'testdata/Beads_63X_NA1.35_xy=0.042_z=0.1.czi'
#filename = r'testdata/T=5_Z=3_CH=2_CZT_All_CH_per_Slice.czi'
filename = r'c:\Users\M1SRH\Documents\Testdata_Zeiss\Castor\Beta_Bonn\20150923\PSF_20X_0.7_2X_488nm_dz=0.25_D153-173.czi'

imgbase = os.path.basename(filename)
imgdir = os.path.dirname(filename)

# specify bioformats_package.jar to use if required
#bf.set_bfpath('c:\Users\M1SRH\Documents\Spyder_Projects\PSF_XYZ_Bead_Fit_BioFormats\bioformats_package.jar')

# get image meta-information
MetaInfo = bf.bftools.get_relevant_metainfo_wrapper(filename)
img6d = bf.bftools.get_image6d(filename, MetaInfo['Sizes'])

## show relevant image Meta-Information
print '\n'
print 'Image Directory      : ', imgdir
print 'Image Filename       : ', imgbase
print 'Images Dim Sizes     : ', MetaInfo['Sizes']
print 'Dimension Order*     : ', MetaInfo['DimOrder BF']
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
print 'Channels             : ', MetaInfo['Channels']
print 'Channel Description  : ', MetaInfo['ChDesc']
print 'Array Shape 6D       : ', np.shape(img6d)

