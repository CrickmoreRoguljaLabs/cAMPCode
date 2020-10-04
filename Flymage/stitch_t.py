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


def dfof_analysis(file_path, t_base=None, F_base = None, channel=0, average=1):
	# compute dfof using the sum of fluorescence in an image (no ROIs in this analysis)
	flimobj = FlimObj.FlimObj()
	flimobj.read_FLIM(file_path)
	flimobj.times_to_datetime() # adds an attribute to the flimobj that stores the times as datetimes (flimobj.datetimes)
	image_array = flimobj.get_images_as_intensity_array()
	x_in_seconds, dfof, F_base, t_base = flimobj.dfof_analysis(image_array,t_base=t_base, F_base=F_base, channel=channel, average=average)

	return x_in_seconds, dfof, flimobj, F_base, t_base

def pulsed_t_series(t_axis, dfof, flimobj,average=1):
	# handle the pulses by turning them into nans, return the times of pulses
	pulse_params = flimobj.FLIMageFileReader.State.Uncaging
	# params (in milliseconds)
	(nPulses, pulsewidth, isi, delay) = (pulse_params.nPulses, pulse_params.pulseWidth, pulse_params.pulseISI, pulse_params.pulseDelay)
	pidx_list = []
	p_offset = pulse_params.baselineBeforeTrain_forFrame
	for p in range(nPulses):
		pstart = delay + p*(isi)+p_offset
		pend = p*(isi)+delay + pulsewidth+ p_offset # in milliseconds
		# now measure from start to end of each pulse
		pidxs = np.arange(flimobj.last_frame_before_time(pstart, units="milliseconds")-1,\
			flimobj.last_frame_before_time(pend, units="milliseconds")+1,dtype=np.int)
		pidxs = (pidxs/average).astype(np.int)
		dfof[pidxs] = np.nan
		pidx_list.append(pidxs)
	return dfof, pidx_list

def single_round_process(filename,plot=False, F_base = None, t_base=None, channel=0, average=1):
	# everything I want to do to a single trace
	t_axis, dfof, flimobj, F_base, t_base = dfof_analysis(filename, F_base=F_base, t_base=t_base, channel=channel, average=average) # the flimobj contains a lot of additional info
	if flimobj.FLIMageFileReader.State.Uncaging.uncage_whileImage:
		dfof, pidxs = pulsed_t_series(t_axis, dfof, flimobj,average=average) # deal with the pulses, find out which frames correspond to them
		if not t_base:
			t_axis = t_axis - t_axis[pidxs.ravel()[0]]
	# now write t and dfof to an excel file
	#writer = pd.ExcelWriter('%s/%s.xlsx'%(dirpath,filename[:-5]), engine='xlsxwriter')
	#pd.DataFrame(np.vstack((t_axis, dfof)).T).to_excel(writer)
	#writer.save()

	if plot:
		plt.plot(t_axis,dfof)
		plt.show()
	return t_axis, dfof, F_base, t_base

def main(argv):
	# extract number of frames to average
	unixOpts = "a:c:s"
	gnuOpts = ["average=","channels=","save="]
	try:
		avg=1
		channels=1
		arguments, values = getopt.getopt(argv, unixOpts, gnuOpts)
		for opt, arg in arguments:
			print(opt,arg)
			if opt in ('-a',"--average"):
				avg = int(arg)
			elif opt in ('-s', '--save'):
				save = True
			elif opt in ('-c', '--channels'):
				channels = int(arg)
	except getopt.GetoptError as err:
		print(str(err))
		usage()
		sys.exit(2)

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

	filelist = sorted(glob.glob("%s*.flim"%(basename)),key=lambda x: int(re.findall("(\d+)",x)[-1])) # sort numerically

	t_list = [[] for c in range(channels)]
	dfof_list = [[] for c in range(channels)]
	F_base = []
	t_base = []

	for file in filelist:
		for c in range(channels): # iterate through color channels
			if not dfof_list[c]: # check if empty
				t_axis, dfof, F_base_c, t_base_c = single_round_process(file, channel=c, average=avg) # process each trace
				F_base.append(F_base_c)
				t_base.append(t_base_c)
			else:
				t_axis,dfof, _,_ = single_round_process(file, F_base=F_base[c], t_base=t_base[c], average=avg, channel=c)
			t_list[c].append(t_axis)
			dfof_list[c].append(dfof)

	t_axis = np.concatenate(t_list[0], axis = 0).flatten()
	dfof = [np.concatenate(dfof_list[c], axis = 0).flatten() for c in range(channels)]
	dfof = np.array(dfof)
	#save = True
	if save:
		concat_trace = pd.DataFrame(np.vstack((t_axis.T,dfof)).T)
		sum_writer = pd.ExcelWriter('%s--stitched.xlsx' %(basename),engine='xlsxwriter')
		concat_trace.to_excel(sum_writer)
		pd.DataFrame(F_base).to_excel(sum_writer,sheet_name="Base fluorescence")
		sum_writer.save()
	for c in range(channels):
		plt.plot(t_axis, dfof[c,:])
	plt.show()
	root.destroy()

if __name__ == "__main__":
	main(sys.argv[1:])