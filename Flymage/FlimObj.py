import matplotlib as mpl
import matplotlib.pyplot as plt
import FLIMageFileReader
import numpy as np
import re, glob
import datetime, os
from Vizier import Vizier
import tifffile as tiff

class FlimObj: # designed to work as an intermediary between the FLIMageFileReader and analyses I like to run
	def __init__(self):
		pass

#################### READING #################################################

	def read_FLIM(self,flim_file):
		""" reads a FLIM file """
		iminfo = FLIMageFileReader.FileReader()
		iminfo.read_imageFile(flim_file, readImage=True)
		self.FLIMageFileReader = iminfo
		self.times_to_datetime() # create an attribute that uses datetime time stamps
		self.filename = flim_file
		self.verbose = False
		self.z_stack = iminfo.ZStack

	def times_to_datetime(self):
		# convert the time stamps into datetimes and store them in a list
		self.datetimes = []
		for times in self.FLIMageFileReader.acqTime:
			self.datetimes.append(datetime.datetime.strptime(times, '%Y-%m-%dT%H:%M:%S.%f'))

	def get_image_file(self):
		# return the image list
		return self.FLIMageFileReader.image

	def get_images_as_intensity_array(self, flimbins = False):
		# output a numpy array that is [t,z,c,x, y, tflim]
		images = self.FLIMageFileReader.image
		t_steps = len(images)
		z_steps = len(images[0])
		c_steps = len(images[0][0])

		if flimbins:
			image_array = np.zeros((t_steps,z_steps,c_steps,self.FLIMageFileReader.height,self.FLIMageFileReader.width,np.max(self.FLIMageFileReader.n_time))) # array of zeros as a placeholder
			for t in range(t_steps):
				for z in range(z_steps):
					for c in range(c_steps):
						image_array[t,z,c,:,:,:] = images[t][z][c]
		else:
			image_array = np.zeros((t_steps,z_steps,c_steps,self.FLIMageFileReader.height,self.FLIMageFileReader.width)) # array of zeros as a placeholder

			for t in range(t_steps):
				for z in range(z_steps):
					for c in range(c_steps):
						this_image = images[t][z][c]
						image_array[t,z,c,:,:]=np.sum(this_image,axis=2)
		if self.z_stack:
			image_array = image_array.swapaxes(0,1)
		return image_array

	def get_pulse_times(self):
		# return the frame indices during which opto stim occurred
		pulse_params = self.FLIMageFileReader.State.Uncaging # contains all the variables we need
		(nPulses, pulsewidth, isi, delay) = (pulse_params.nPulses, pulse_params.pulseWidth, pulse_params.pulseISI, pulse_params.pulseDelay)
		pidx_list = []
		for p in range(nPulses):
			pstart = delay + p*(isi)
			pend = p*(isi+pulsewidth)+delay + pulsewidth # in milliseconds
			# now measure from start to end of each pulse
			pidxs = range(self.last_frame_before_time(pstart, units="milliseconds")-1,\
				self.last_frame_before_time(pend, units="milliseconds"))
			pidx_list.append(pidxs)
		return pidx_list

	def get_pulse_delay(self, units="seconds"):
		# returns delay to first uncaging pulse
		delay = self.FLIMageFileReader.State.Uncaging.pulseDelay # in milliseconds
		try:
			if units == "seconds":
				delay=delay/1000.0
			if units == "minutes":
				delay = delay/(60000.0)
			if units == "milliseconds":
				delay = delay
		except:
			raise Exeception("Delay units incorrect. Use 'seconds','minutes', or 'milliseconds' (Default is seconds)")
		return delay

	def get_pulse_frames_baseline(self):
		return self.FLIMageFileReader.State.Uncaging.FramesBeforeUncage

	def get_pulse_width(self, units="seconds"):
		# returns width of first uncaging pulse
		pulse = self.FLIMageFileReader.State.Uncaging.pulseWidth # in milliseconds
		try:
			if units == "seconds":
				pulse=pulse/1000.0
			if units == "minutes":
				pulse = pulse/(60000.0)
			if units == "milliseconds":
				pulse = pulse
		except:
			raise Exeception("Pulse units incorrect. Use 'seconds','minutes', or 'milliseconds' (Default is seconds)")
		return pulse

	def last_frame_before_time(self, moment, units="seconds"):
		 # returns the last frame before the event described as "moment" in units "units"
		t_axis = self.get_t_axis(units=units)
		return np.sum((t_axis <= np.floor(moment)))

##################### ANALYSIS ########################

	### DFOF of an ROI
	def dfof_analysis(self,image_array, t_base=None, F_base = None, channel=0, average=1):
		# needs an image array to compute dfof in
		x_in_seconds = self.get_t_axis() # it's in seconds unless you specify otherwise
		pulse_t = self.get_pulse_delay()

		# find when pulses occur
		pulse_frame = int(np.sum(x_in_seconds<pulse_t) + self.get_pulse_frames_baseline())
		pulse_frame = int(pulse_frame / average)

		# pool all the pixels in the ROI
		summed_vals = np.sum(image_array,(3,4))
		frames = summed_vals.shape[0]
		# if averaging multiple frames:
		if average > 1:
			summed_vals = np.add.reduceat(summed_vals,np.arange(0,frames,average),axis=0) # sum merged images
			if frames % average > 0:
				summed_vals[-1] = summed_vals[-1]/(np.float(frames%average)/np.float(average)) # account for pooled parts that don't contain the full "average"
			x_in_seconds = x_in_seconds[::average]# take first time point
		if F_base:
			pass
		else: # define the F0 value -- the average of the 5 frames preceding the first pulse
			if pulse_frame < 5:
				F_base = np.mean(summed_vals[0:5,0,channel,...])
			else:
				F_base = np.mean(summed_vals[(pulse_frame-5):(pulse_frame-1),0,channel])
		dfof = (summed_vals[:,0,channel]-F_base)/F_base
		if t_base:
			t_offset = (self.datetimes[0]-t_base).total_seconds()
			x_in_seconds = x_in_seconds+t_offset
		else:
			x_in_seconds = x_in_seconds-pulse_t
			try:
				t_base = self.datetimes[pulse_frame]
			except:
				print("Pulse parameters outside of acquired data! Double check to make sure analysis is right.")
				t_base = self.datetimes[0]
		if self.z_stack:
			x_in_seconds = np.array([0])
		return x_in_seconds, dfof, F_base, t_base

	def emp_tau(self,x, tauo=None):
		# x is a vector of photons and flim bins
	    tpu = self.FLIMageFileReader.State.Spc.spcData.resolution[0]
	    tbins = np.arange(x.shape[0])
	    if not tauo:
	        from scipy.optimize import minimize

	        params = np.array([0.5,10.5,3.0,1.0,15.0]) # just some guess
	        from scipy.optimize import Bounds
	        bounds = Bounds([0.0,1.0,1.0,0.0,0.0],[1.0,np.inf,np.inf,np.inf,np.inf])

	        res = minimize(self.nll,params,args=(x,),method='trust-constr', bounds=bounds) # maybe one day I will Bayes it up in here.
	        if self.verbose:
	        	print(res)
	        tauo=res.x[-1]
	        self.flim_params = res.x
	        print(res.x)
	    return [(np.nansum(x*(tbins>=(tauo))*(tbins-(tauo))*tpu)/np.nansum(x*(tbins>=(tauo))))/1000.0, tauo] # in nanoseconds

################### Visualization ################################

	def viz_t_stack(self,im_array=None):
		""" Make a window corresponding to the pixelwise lifetime map across time to be scrolled through
		"""
		viz = Vizier.Vizier() # toy image viewer

		try:
			viz.image_from_array(im_array)
		except:
			print("Error importing image into array")
		#plt.imshow(self.t_stack[:,:,40])
		#plt.show()
		viz.viz_images(scroll_label="t step")

	def viz_frame(self, im_array,axes_dict={'z':0,'x':1,'y':2,'c':3,'t':4}, color=0):
		## Show a single frame at a time, return the image_scroller object created
		viz = Vizier.Vizier()
		viz.image_from_array(im_array,axes_dict=axes_dict)
		scr = viz.show_image(color=color)
		return scr

		#### MAKE A TIFF FILE

	def flimtiff(self, flim=False):
		if not self.tauo:
			raise Exception("Offset tau not estimated for this data")

		# useful values / arrays

		tauo = self.tauo
		avg = self.avg
		im_array = self.get_images_as_intensity_array(flimbins=flim)

		from scipy.signal import convolve
		avged = np.add.reduceat(im_array,np.arange(0,im_array.shape[0],avg),axis=0)
		#avged = convolve(avged,np.ones((1,1,1,3,3,1),dtype=int),'valid')
		tbins = np.arange(avged.shape[-1])
		tpu = self.FLIMageFileReader.State.Spc.spcData.resolution[0] # time per flim bin
		min_val = 1*avg
		## compute empirical tau for each pixel using the saved "avged" object
		intensity = np.sum(avged,axis=-1)
		px_tau = (intensity>=min_val)*np.sum((avged*(tbins>=tauo)*(tbins-tauo)*tpu),axis=-1)/np.sum(avged*(tbins>=tauo),axis=-1) #px_tau in picoseconds
		px_tau = px_tau.astype(np.uint32)
		tiff.imsave("%s.tiff"%(os.path.splitext(self.filename)[0]),px_tau)
		tiff.imsave("%s_intensity.tiff"%(os.path.splitext(self.filename)[0]),intensity)

#################### FITTING ###################################

	def fit_tau(self, im_array, t_base = None, avg=1, tauo=None, channel=0):
		# fits each image to a biexponential decay model, return the model params
		# Dimensions are [time, z, color] if px_wise false, [time, z, color, x, y] if px_wise true
		self.avg = avg
		img_fused = np.sum(im_array,axis=(3,4)) # combine all pixels
		avged = np.add.reduceat(img_fused,np.arange(0,img_fused.shape[0],avg),axis=0) # sum merged images
		merged_t = self.get_t_axis()[::avg]# take first time point
		tpu = self.FLIMageFileReader.State.Spc.spcData.resolution # time per bin
		if not tauo: # see if I've already fit this data to an offset
			_, tauo = self.emp_tau(np.sum(avged[:,:,channel,:],axis=(0,1)))
		emp_taus = np.array([self.emp_tau(np.sum(avged[k,:,channel,:],axis=0),tauo=tauo)[0] for k in range(avged.shape[0])])
		self.tauo=tauo
		return merged_t, emp_taus

	#### FLIM FITTING

	def prob_bin(self, params,k):
	    # probability of observing a photon in time bin k
	    # function is in units of the time bins of the photon counter
	    # f = proportion in tau 1 state
	    # tau 1 = first time constant
	    # tau 2 = second time constant
	    # taug = Gaussian IRF
	    # tao = time offset
	    from scipy.special import erfc
	    f = params[0]
	    tau1 = params[1]
	    tau2 = params[2]
	    taug = params[3]
	    tauo = params[4]
	    
	    
	    k = k + (self.get_tflim_num())*(k<tauo) # for the pre-pulse part, wrap them around by adding the "post" bins
	    decay1 = f*np.exp(-k/tau1)*np.exp(tauo/tau1)*np.exp((taug**2.0)/(2.0*tau1**2.0))*(1.0/tau1) # monoexponential decay
	    irf1 = erfc((taug**2.0 - tau1*(k-tauo))/(np.sqrt(2.0)*tau1*taug)) # convolutional integral
	    decay2 = (1.0-f)*np.exp(-k/tau2)*np.exp(tauo/tau2)*np.exp((taug**2.0)/(2.0*tau2**2.0))*(1.0/tau2)
	    irf2 = erfc((taug**2.0 - tau2*(k-tauo))/(np.sqrt(2.0)*tau2*taug)) # convolutional integral

	    prob_val = decay1*irf1/2.0 + decay2*irf2/2.0 # dropped the 2, oops.
	    #prob_val = np.sqrt(np.pi * taug)*(f*np.exp(-k/tau1 +tauo/tau1+taug/(4*tau1**2))+(1-f)*np.exp(-k/tau2 +tauo/tau2+taug/(4*tau2**2)))
	    
	    return prob_val

	def nll(self,params,x):
	    # negative log-likelihood of binned photon count data x
	    f=params[0]
	    tau1=params[1]
	    tau2=params[2]
	    taug=params[3]
	    tauo=params[4]
	    tbins = np.arange(x.shape[0])
	    return -np.nansum(x[:-1]*np.log(self.prob_bin(params,tbins[:-1])))

	#### BASIC PARAMETERS OF THE DATA

	def get_t_axis(self, units="seconds", datetime = False):
		# returns t_axis
		if datetime:
			return self.datetimes
		else:
			try:
				if units == "seconds":
					t_axis = [(T - self.datetimes[0]).total_seconds() for T in self.datetimes]
				if units == "minutes":
					t_axis = [(T - self.datetimes[0]).total_seconds()/60.0 for T in self.datetimes]
				if units == "milliseconds":
					t_axis = [(T - self.datetimes[0]).total_seconds()*1000.0 for T in self.datetimes]
			except:
				raise Exeception("Delay units incorrect. Use 'seconds','minutes', or 'milliseconds' (Default is seconds)")
			return np.array(t_axis)

	def get_width(self):
		return self.FLIMageFileReader.width

	def get_height(self):
		return self.FLIMageFileReader.height

	def get_tflim_num(self):
		time_bins = self.FLIMageFileReader.n_time
		if not (time_bins[0] == time_bins[1]):
			raise Exception("FLIM channels do not use same number of time bins, use other analysis code (sorry!)." \
				"\nIf you encounter this error, please inform SCT.")
		return time_bins[0]