# for batch processing of t series that are similar
import FlimObj
from tkinter import filedialog
import re, glob, os
import matplotlib.pyplot as plt
import datetime
import numpy as np
import tkinter as Tk
import pandas as pd
import sys, getopt

# I KNOW I KNOW. but the solver starts to make a lot of noise when the estimates start to converge.
import warnings
warnings.filterwarnings("ignore")

def roi_analysis(flimobj,roi_im,tauo=None):
	if tauo:
		_, emp_taus = flimobj.fit_tau(roi_im,tauo=tauo, avg=1)
	else:
		_,emp_taus = flimobj.fit_tau(roi_im,avg=1)

	return emp_taus

def flim_analysis(file_path, rois = None, t_base=None, tauo = None, channel=0, average=1):
	# compute dfof for each ROI
	## copied from stitch_t.py
	flimobj = FlimObj.FlimObj()
	flimobj.read_FLIM(file_path)
	flimobj.times_to_datetime() # adds an attribute to the flimobj that stores the times as datetimes (flimobj.datetimes)
	image_array = flimobj.get_images_as_intensity_array(flimbins=True)
	frames = image_array.shape[0]
	image_array = np.add.reduceat(image_array,np.arange(0,frames,average),axis=0) # sum merged images
	print(image_array.shape)
	# fit tauo if none
	if not tauo:
		flimobj.fit_tau(np.sum(image_array,axis=1,keepdims=True),avg=1,channel=channel)
		tauo = flimobj.tauo

	## Fix the t axis if there's a pulse
	x_in_seconds = flimobj.get_t_axis() # it's in seconds unless you specify otherwise
	#pulse_t = flimobj.get_pulse_delay()

	xmax = image_array.shape[3]
	ymax = image_array.shape[4]
	


	# pick the time point when averaging
	if average > 1: # pool together time bins if averaging time bins
		x_in_seconds = x_in_seconds[::average]# take first time point

	## ROI SPECIFIC ANALYSIS HERE
	# rois is a list of tuples of the x and y bounds of each ROI, and the z value. Structure: [(roi,z),(roi,z)]. Structure of roi: ((x0,y0),(x1,y1))
	emp_taus_list = []

	for roi_pair in rois:
		roi, z_val = roi_pair[0], roi_pair[1]
		xbounds = np.sort([np.min((roi[0][0],xmax)),np.min((roi[1][0],xmax))])
		ybounds = np.sort([np.min((roi[0][1],ymax)),np.min((roi[1][1],ymax))]) # make sure the bounds don't get messed up
		print("Evaluating ROI!")
		roi_im = np.expand_dims(image_array[:,z_val,:,ybounds[0]:ybounds[1],xbounds[0]:xbounds[1],...],axis=1) # UGH the axis reversal
		print(roi_im.shape)
		emp_taus = roi_analysis(flimobj,roi_im,tauo=tauo) # apply analysis to the ROI
		#print(emp_taus)
		emp_taus_list.append(emp_taus)


	return x_in_seconds, emp_taus_list, tauo, rois, flimobj

def pulsed_t_series(t_axis, dfof, flimobj,average=1):
	# handle the pulses by turning them into nans, return the times of pulses
	pulse_params = flimobj.FLIMageFileReader.State.Uncaging
	# params (in milliseconds)
	(nPulses, pulsewidth, isi, delay) = (pulse_params.nPulses, pulse_params.pulseWidth, pulse_params.pulseISI, pulse_params.pulseDelay)
	pidx_list = []
	p_offset = pulse_params.baselineBeforeTrain_forFrame
	for p in range(nPulses):
		pstart = delay + p*(isi)+p_offset
		pend = p*(isi+pulsewidth)+delay + pulsewidth+p_offset # in milliseconds
		# now measure from start to end of each pulse
		pidxs = np.floor(np.arange(flimobj.last_frame_before_time(pstart, units="milliseconds")-1,\
			flimobj.last_frame_before_time(pend, units="milliseconds"),average)/average).astype(int)
		for j in range(len(dfof)):
			dfof[j][pidxs] = np.nan
		pidx_list.append(pidxs)
	return dfof, pidx_list

def single_round_process(filename,plot=False, tauo=None, t_base=None, channel=0, average=1, rois = None):
	# everything I want to do to a single trace
	t_axis, emp_taus, tauo, rois, flimobj = flim_analysis(filename, tauo=tauo, t_base=t_base, channel=channel, average=average, rois = rois) # the flimobj contains a lot of additional info
	
	# if there are pulses
	if flimobj.FLIMageFileReader.State.Uncaging.uncage_whileImage:
		emp_taus, pidxs = pulsed_t_series(t_axis, emp_taus, flimobj,average=average) # deal with the pulses, find out which frames correspond to them
		if not t_base:
			t_axis = t_axis - t_axis[pidxs[0].ravel()[0]]
			t_base = t_axis[pidxs[0].ravel()[0]]
	else:
		t_base = t_axis[0]
	if plot:
		for roi in emp_taus:
			plt.plot(t_axis,np.array(roi))
	return t_axis, emp_taus,tauo, t_base

def main(argv):
	# extract number of frames to average, other params
	unixOpts = "a:c:sz"
	gnuOpts = ["average=","channels=","save=","zstack="]
	try:
		avg=1
		channels=1
		save=False
		z_stack = False
		arguments, values = getopt.getopt(argv, unixOpts, gnuOpts)
		for opt, arg in arguments:
			print(opt,arg)
			if opt in ('-a',"--average"):
				avg = int(arg)
			elif opt in ('-s', '--save'):
				save = True
			elif opt in ('-c', '--channels'):
				channels = int(arg)
			elif opt in ('-z', '--zstack'):
				z_stack = True
	except getopt.GetoptError as err:
		print(str(err))
		usage()
		sys.exit(2)

	# GUI stuff
	root = Tk.Tk()
	root.withdraw()
	#direc = filedialog.askdirectory()
	root.update()
	file_path = filedialog.askopenfilename()
	root.update()
	try:
		basename = re.match("(.*\D)[0-9]+\.flim",file_path).group(1)  # up to the part that ends in <numbers>.flim
	except: # file not selected
		sys.exit(0)

	### find files

	filelist = sorted(glob.glob("%s*.flim"%(basename)),key=lambda x: int(re.findall("(\d+)",x)[-1])) # sort numerically

	### preparatory arrays for the time axis, the tau axis, etc.

	t_list = [[] for c in range(channels)]
	tau_list = [[] for c in range(channels)]
	tauo_list = [None for c in range(channels)]
	t_base = [None for c in range(channels)]

	### Define rois
	roi_obj = FlimObj.FlimObj()
	roi_obj.read_FLIM(filelist[0]) # select roi from first frames
	image_array = roi_obj.get_images_as_intensity_array(flimbins=False)
	print(image_array.shape)
	# contains an attribute roi_coords with the coordinates for each ROI
	if channels==2:
		image_scroller = roi_obj.viz_frame(image_array,axes_dict={'t':0,'z':1,'c':2,'x':3,'y':4},color=1)
	else:	
		image_scroller = roi_obj.viz_frame(image_array,axes_dict={'t':0,'z':1,'c':2,'x':3,'y':4})
	roi_coords = image_scroller.roi_coords # format is list of length 1:z_max containing roi for each z plane in format ((x0,y0),(x1,y1))
	rois = [(roi,z) for z in range(len(roi_coords)) for roi in roi_coords[z]]

#	Try it for yourself, it's the stupid index reversal. Not sure why it works that way.
#	xmax = image_array.shape[3]
#	ymax = image_array.shape[4]
#	for pair in rois:
#		roi, z = pair[0], pair[1]
#		xbounds = np.sort([np.min((roi[0][0],xmax)),np.min((roi[1][0],xmax))])
#		ybounds = np.sort([np.min((roi[0][1],ymax)),np.min((roi[1][1],ymax))]) # make sure the bounds don't get messed up
#		plt.figure()
#		plt.imshow(image_array[0,z,0,ybounds[0]:ybounds[1],xbounds[0]:xbounds[1]])
#	plt.show()

 	### iterate through each file, perform analysis

	for file in filelist:
		for c in range(channels): # iterate through color channels
			if not tauo_list[c]: # check if empty
				t_axis, emp_taus, tauo, tb = single_round_process(file, channel=c, average=avg, rois=rois,plot=False) # process each trace
				tauo_list[c] = tauo
				t_base[c] = tb
			else:
				t_axis,emp_taus, _,_ = single_round_process(file, tauo=tauo_list[c], t_base=t_base[c], average=avg, rois=rois, plot=False)
			t_list[c].append(t_axis)
			tau_list[c].append(emp_taus)

	### merging all the arrays from each individual file
	if not z_stack:
		t_axis = [np.concatenate(t_list[c], axis = 0).flatten() for c in range(channels)]
		t_axis = t_axis[0]
	taus = [np.concatenate(tau_list[c], axis = 1).T for c in range(channels)] # dimensions: time point x cells
	#taus = np.array(taus)
	#save = True
	if save:
		sum_writer = pd.ExcelWriter('%s--stitched_rois.xlsx' %(basename),engine='xlsxwriter')
		for c in range(channels):
			if not z_stack:
				concat_trace = pd.DataFrame(np.hstack((t_axis[:,np.newaxis],taus[c])))
			else:
				concat_trace = pd.DataFrame(taus[c])
			concat_trace.to_excel(sum_writer, sheet_name="Color %s"%(c))
			pd.DataFrame(rois).to_excel(sum_writer,sheet_name="Rois")
			sum_writer.save()
	#for c in range(channels):
		#plt.plot(t_axis, dfof[c,:])
	plt.show()
	root.destroy()

if __name__ == "__main__":
	main(sys.argv[1:])