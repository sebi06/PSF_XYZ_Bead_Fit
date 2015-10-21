===============================
PSF-XYZ Bead Fit
===============================

This program can be used to detect beads an measures the FWHM-XYZ of the PSF.
The crucial steps are:

* read the z-stack via BioFormats
* find the brightest voxel and uses the corresponding XY-plane (hot pixels still may ruin the results)
* detect all peaks within the XY-plane from 2)
* extract the Z-profile at every detected peak position
* determine the brightest XY-plane for every peak separately
* cutout sub-images at every detected peak position
* do 2D-Gauss fit for every peak to determine FWHM-XY
* do 1D-Gauss fit for every Z-Profile to determine FWHM-Z
* displays PSF OrthoView and PSF volume for the average PSF = sum of all detected PSFs
* optional - displays PSF OrthoView and PSF volume for one randomly selected peak (optional)
* optional - write results to excel sheet (currently only XLS)
* optional - save output graphics as PNGs

:Author: Sebastian Rhode

:Version: 2015.10.21

Requirements
------------
* `CPython 2.7 <http://www.python.org>`_
* `Numpy 1.8.2 <http://www.numpy.org>`_
* `xlwt <https://pypi.python.org/pypi/xlwt>`_
* `SciPy <https://pypi.python.org/pypi/scipy>`_
* `VisVis <https://pypi.python.org/pypi/visvis>`_
* `Scikit-Image <https://pypi.python.org/pypi/scikit-image>`_
* `bfimage package <https://github.com/sebi06/BioFormatsRead>`_

Notes
-----
The package is still under development and was mainly tested with CZI files. Fell free to improve it and give feedback.
It uses BioFormats to read the image data into numpy arrays and uses the package bfimage to read the actual image data.

References
----------
(1)  CZI - Carl Zeiss Image format for microscopes - http://www.zeiss.com/czi
(2)  The OME-TIFF format - http://www.openmicroscopy.org/site/support/file-formats/ome-tiff
(3)  Read microscopy images to numpy array with python-bioformats - http://ilovesymposia.com/2014/08/10/read-microscopy-images-to-numpy-arrays-with-python-bioformats/
(4)  The Python-BioFormats package - http://downloads.openmicroscopy.org/bio-formats/
(5)  czifile.py by Christoph Gohlke - http://www.lfd.uci.edu/~gohlke/code/czifile.py.html
(6)  tifffile.py by Christoph Gohlke - http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html

Screenshots
-----------

Main GUI:

.. figure:: images/PSF-XYZ_GUI.png
   :align: center
   :alt: 

FHWM Report:

.. figure:: images/PSF_XYZ_FWHM.png
   :align: center
   :alt: 

PSF Volume & Cross Sections:

.. figure:: images/PSF-XYZ_Volume_Cross.png
   :align: center
   :alt: 

PSF OrthoView:

.. figure:: images/PSF-XYZ_OrthoView.png
   :align: center
   :alt:
