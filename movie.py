# Supplementary material of the PhD thesis of Johannes Mielke (johannes-mielke@gmx.de)
# submitted to the Physics department of the Freie Universität Berlin in 2013 
# and carried out at the Fritz-Haber-Institut of the Max-Planck-Gesellschaft
#
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.ndimage
import scipy as sp

class movie:
	"""The movie class can load a movie from a series of .npy files. Most of the processing is done in the constructor, there the images can be
	height corrected and hot pixels can be removed. After that, the drift of the images can be determined and corrected in the following. An average
	image is also calculated.
	
	Some methods show frames or their differences and functions are present to export the whole movie to png.
	The heights of pixels can also be exported
	"""
	def __init__(self,basename,n_images,s_images,images=[],shift=True,hotremove=1,mean_corr=True,sequential=False,blur=1,keyframes=[],avg_except=[],fitdrift=False):
		
		self.n_images = n_images	#number of images in this movie
		self.s_images = s_images	#size in Angstrom of a frame
		self.basename = basename	#basename of the movie
		self.imshift = []	#shift values for every frame
		self.shape=[]	#shape of the shifted images
		self.hotremove = hotremove
		self.blur = blur
		self.mean_corr = mean_corr
		self.avg_except = avg_except #The images in this list are filled with zeros which has the effect
									 #that all the image counts as "out of image" and concerning the time
									 #dependence the movie is effectively split in parts between the 
									 #excluded images
		

			
		if shift:
			try:
				self.imshift = np.load(basename+"_imshift.npy")
			except IOError:
				#First determine the shift in between the images
				if sequential:
					print "Determining the shift of all images with respect to the previous one ..."
					self.imshift = self.determine_series_shift_sequential()
				else:
					if keyframes==[]:
						print "Determining the shift of all images with respect to the first ..."
						self.imshift = self.determine_series_shift()
					else:
						print "Determining the shift using the keyframes: ", keyframes 
						self.imshift = self.determine_series_shift_keyframes(keyframes)
				if fitdrift: self.fitdrift()
	
		
		#create an average image of the movie if not yet existing
		try:
			self.avgim = np.load(basename+"_avgim.npy")
		except IOError:
			print "Creating average image"
			self.avgim=self.load_image_shifted(0)
			for i in range(1,self.n_images):
				self.avgim = self.avgim + self.load_image_shifted(i)
	
		self.shape = self.avgim.shape
		self.org_shape = self.load_image(0).shape

		if self.imshift != []:
			np.save(self.basename+"_imshift.npy",self.imshift)
		np.save(self.basename+"_avgim.npy",self.avgim)


	def load_image_shifted(self,num):
		if self.imshift == []:
	    		print "Shift not calculated yet!"
	  	else:
			image = self.load_image(num)
			maxshift = [np.max(self.imshift[:,0]),np.max(self.imshift[:,1])]
			minshift = [np.min(self.imshift[:,0]),np.min(self.imshift[:,1])]
			print "Shifting the image"
			simage = self.image_shift(image,self.imshift[num],minshift,maxshift)
			return simage



	def load_image(self,num):
		print "Loading image "+str(num+1)+" from disk"
		image = np.load(self.basename + "_" + str(num+1) + ".npy")
		print "Done"	
		if num in self.avg_except:
			print "Replacing image with zeros"
			image = np.zeros(image.shape)
		
		#removing the "hotremove brightest pixels from the images"
		print "Removing bright and dark pixels ..."		
		for k in range(self.hotremove):			
			coord = np.unravel_index(np.argmax(image),image.shape)
			image[coord[0],coord[1]]=np.median(image)
			coord = np.unravel_index(np.argmin(image),image.shape)
			image[coord[0],coord[1]]=np.median(image)
		print "Bright and dark pixels removed"
	
		if self.blur >1:
			print "Bluring the image..."
			image = map(lambda x: scipy.ndimage.convolve(x,np.ones([self.blur,self.blur])),image)

		#now remove the mean value from all images
		if self.mean_corr:
			print "Subtract median from image ..."
			image=image-np.median(image)
			print "Median subtracted from images"
			
		return image



	def determine_shift_cross_fft(self,im1,im2):
		"""This function calculates the shift between two images, by calculating the cross correlation using an fft algorithm"""
		#calculate fourier transormed image
		sim1 = fft.fft2(im1)
		sim2 = fft.fft2(im2)
		
		#calculate cross corellation function
		cc = fft.ifft2(sim1*np.conj(sim2))
		
		#look for maximum and give that out
		[x,y]=self.locate_max(cc)
		x=-x
		y=-y
		if x<-len(im1)/2:
			x = x+len(im1)
		if y<-len(im1)/2:
			y = y+len(im1)
		return [x,y]	
	
	def determine_series_shift(self):
		"""This function determnes the drift in a series of images by comparing all images with the first one"""
		disp=[np.array([0,0])]
		imref = self.load_image(0)
		for i in range(1,self.n_images):
			disp.append(np.array(self.determine_shift_cross_fft(imref,self.load_image(i))))
			print "Calculate shift: Completed ", i, " of ", self.n_images
		return np.array(disp)

	def determine_series_shift_sequential(self):
		"""This function determines the drift in a series of images by comparing all images with the previous one"""
		disp=[np.array([0,0])]
		
		for i in range(1,self.n_images):
			disp.append(disp[-1] + np.array(self.determine_shift_cross_fft(self.load_image(i-1),self.load_image(i))))
			print "Calculate shift: Completed ", i, " of ", self.n_images
		return np.array(disp)
		
	def determine_series_shift_keyframes(self,keyframes):
		"""This function determnes the drift in a series of images by comparing all images with keyframe before"""
		disp=[np.array([0,0])]
		key = 0
		imref = self.load_image(key)
		for i in range(1,self.n_images):
			disp.append(disp[key] + np.array(self.determine_shift_cross_fft(imref,self.load_image(i))))
			if i in keyframes: 
				key = i				
				imref = self.load_image(key)
			print "Calculate shift: Completed ", i, " of ", self.n_images
		return np.array(disp)
		
	def image_shift(self,im, shift, minshift, maxshift):
		"""This movie pads an image with the appropriate zeros"""
		pad1=np.zeros([maxshift[0]-shift[0],im.shape[1]])
		pad2=np.zeros([shift[0]-minshift[0],im.shape[1]])
		im2 = np.vstack([pad1,im,pad2])
		pad3=np.zeros([im2.shape[0],-minshift[1]+shift[1]])
		pad4=np.zeros([im2.shape[0],maxshift[1]-shift[1]])
		return np.hstack([pad4,im2,pad3])
		
	def locate_max(self,im):
		"""This function returns the coordinates of the brightest pixel in an image"""
		return np.unravel_index(np.argmax(im),im.shape)
		
	def export_png(self,ang=True):
		print "Exporting movie to png: Filename: "+self.basename+"_n.png"
		for i in range(self.n_images):
			plt.figure(1)
			plt.clf()
			if ang == False:
				plt.imshow(self.load_image_shifted(i),cmap=plt.cm.gray)
				plt.xlabel("X [px]")
				plt.ylabel("Y [px]")
			else:
				plt.imshow(self.load_image_shifted(i),cmap=plt.cm.gray,extent=(0,self.shape[1]*self.s_images/self.org_shape[-1],self.shape[0]*self.s_images/self.org_shape[-1],0))
				plt.xlabel("X [$\AA$]")
				plt.ylabel("Y [$\AA$]")
			plt.title("Movie "+self.basename + "\n#" + str(i))
			plt.show()
			plt.savefig(self.basename+"_%06d.png" %i)
		print "Done"
		
	def export_diff_png(self,ang=True):	
		print "Exporting difference movie to png: Filename: "+self.basename+"_n.png"
		for i in range(self.n_images-1):
			plt.figure(1)
			plt.clf()
			if ang == False:
				plt.imshow(self.load_image_shifted(i+1)-self.load_image_shifted(i))
				plt.xlabel("X [px]")
				plt.ylabel("Y [px]")
			else:
				plt.imshow(self.load_image_shifted(i+1)-self.load_image_shifted(i),extent=(0,self.shape[1]*self.s_images/self.org_shape[-1],self.shape[0]*self.s_images/self.org_shape[-1],0))
				plt.xlabel("X [$\AA$]")
				plt.ylabel("Y [$\AA$]")
			plt.title("Difference of Movie "+self.basename + "\n#" + str(i+1)+" - #"+str(i))
			plt.show()
			plt.savefig(self.basename+"_diff_%06d.png" %i)
		print "Done"
		
	def show_frame(self,num,newfig=False,color=plt.cm.gray,comment="",ang=False):
		"""Plot a frame from the movie"""
		if newfig: plt.figure()
		plt.clf()
		if ang == False:
			plt.imshow(self.load_image_shifted(num),cmap=color)
			plt.xlabel("X [px]")
			plt.ylabel("Y [px]")
		else:
			plt.imshow(self.load_image_shifted(num),cmap=color,extent=(0,self.shape[1]*self.s_images/self.org_shape[-1],self.shape[0]*self.s_images/self.org_shape[-1],0))
			plt.xlabel("X [$\AA$]")
			plt.ylabel("Y [$\AA$]")
		plt.title("Movie "+self.basename + "\n#" + str(num) + " " + comment)
		plt.show()
		
	def show_avgim(self,newfig=False,color=plt.cm.gray,comment="",ang=False,clear=True):
		"""This function displays the average image of the movie"""
		if newfig: plt.figure()
		if clear: plt.clf()
		if ang == False:
			plt.imshow(self.avgim,cmap=color)
			plt.xlabel("X [px]")
			plt.ylabel("Y [px]")
			plt.xlim([0,self.shape[1]])
			plt.ylim([self.shape[0],0])
		else:
			plt.imshow(self.avgim,cmap=color,extent=(0,self.shape[1]*self.s_images/self.org_shape[-1],self.shape[0]*self.s_images/self.org_shape[-1],0))
			plt.xlabel("X [$\AA$]")
			plt.ylabel("Y [$\AA$]")
		
		plt.title("Movie "+self.basename + "average\n " + comment)
		plt.show()
	
	def show_diff(self,im1,im2,newfig=False,comment="",ang=False):
		"""This function shows the difference between image1 and image2"""
		if newfig: plt.figure()
		plt.clf()
		if ang == False:
			plt.imshow(self.load_image_shifted(im1)-self.load_image_shifted(im2))
			plt.xlabel("X [px]")
			plt.ylabel("Y [px]")
		else:
			plt.imshow(self.load_image_shifted(im1)-self.load_image_shifted(im2),extent=(0,self.shape[1]*self.s_images/self.org_shape[-1],self.shape[0]*self.s_images/self.org_shape[-1],0))
			plt.xlabel("X [$\AA$]")
			plt.ylabel("Y [$\AA$]")
		plt.title("Difference of Movie "+self.basename + "\n#" + str(im1)+" - #"+str(im2) + " " + comment)
		plt.show()

	def extract_median_height(self, coord, radius):
		"""This function extracts the median height of a circle around coord from images"""
		ec=[coord+np.array([x,y]) for x in range(-radius,radius+1) for y in range(-radius,radius+1) if np.sqrt(x*x+y*y)<radius]
		return np.median([self.simages[:,a[1],a[0]] for a in ec],axis=0)
	
	def extract_median_height_frame(self, frame_num, coord, radius):
		"""This function extracts the median height of a circle around coord from an image"""
		retlist = []
		image = self.load_image_shifted(frame_num)
		for c in coord:
		  retlist.append(np.median([image[a[1],a[0]] for a in c]))
		return np.array(retlist)




	def plot_autocorrelation(self,frame_num,ang=False,newfig=True,color=plt.cm.gray):
		ftim = fft.fft2(self.load_image(frame_num))
		cc = fft.ifft2(ftim*np.conj(ftim)).real
		if newfig: plt.figure()
		plt.clf()
		if ang == False:
			plt.imshow(cc,cmap=color)
			plt.xlabel("X [px]")
			plt.ylabel("Y [px]")
		else:
			plt.imshow(cc,cmap=color,extent=(0,self.shape[1]*self.s_images/self.org_shape[-1],self.shape[0]*self.s_images/self.org_shape[-1],0))
			plt.xlabel("X [$\AA$]")
			plt.ylabel("Y [$\AA$]")
		plt.title("Autocorrelation of frame "+str(frame_num))
		plt.show()

	def fitdrift(self):
		"""	This function fits the shift from self.imshift with a polynomial of second order and owerwrites
			self.imshift with the result. This can be used to smooth the shift if there are no jumps present"""
		if self.imshift != []:
			plt.figure()
			plt.clf()
			xx = np.arange(self.imshift.shape[0])
			plt.plot(xx,self.imshift)
			
			fitfunc = lambda p, x: p[0] + p[1] * x + p[2] * x * x + p[3] * x * x * x + p[4]* np.sqrt(x)
			errfunc = lambda p, x, y: (y - fitfunc(p, x))
			pinit = [0.0, 0.1, 0.01,0.01,0.01]
			out0 = sp.optimize.leastsq(errfunc, pinit,args=(xx, self.imshift[:,0]), full_output=1)
			out1 = sp.optimize.leastsq(errfunc, pinit,args=(xx, self.imshift[:,1]), full_output=1)
			pfinal0 = out0[0]
			pfinal1 = out1[0]
			yy0 = np.round(fitfunc(pfinal0,xx))
			yy1 = np.round(fitfunc(pfinal1,xx))
			
			plt.plot(xx,yy0,label="xshift (fit)")
			plt.plot(xx,yy1,label="yshift (fit)")
			plt.grid()
			plt.legend(loc=0)
			plt.xlabel("Frame")
			plt.ylabel("Shift [px]")
			plt.title("Measured shift and fited version")
			plt.show()
			self.imshift = np.vstack((yy0,yy1)).transpose()
			
		else:
			print "No shift defined yet!"
		