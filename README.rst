===============================
PSF_XYZ_Bead_Fit
===============================

This package can be used to read image data recorded with beads to check the PSF.
It uses BioFormats to read the image data into numpy arrays.

:Author: Sebastian Rhode

:Version: 2015.07.31

Requirements
------------
* `CPython 2.7 <http://www.python.org>`_
* `Numpy 1.8.2 <http://www.numpy.org>`_
* `bfimage package <https://github.com/sebi06/BioFormatsRead>`_

Notes
-----
The package is still under development and was mainly tested with CZI files.

Acknowledgements
----------------
*   Christoph Gohlke from providing the czifily.py.
*   The Cellprofiler team for providing python-bioformats
*   The OME people for creating BioFormats                                                                                 

References
----------
(1)  CZI - Image format for microscopes
     http://www.zeiss.com/czi
(2)  The OME-TIFF format.
     http://www.openmicroscopy.org/site/support/file-formats/ome-tiff
(3)  Read microscopy images to numpy array with python-bioformats.
     http://ilovesymposia.com/2014/08/10/read-microscopy-images-to-numpy-arrays-with-python-bioformats/
