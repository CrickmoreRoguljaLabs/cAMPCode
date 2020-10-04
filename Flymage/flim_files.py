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


def flim_analysis(file_path, avg=1, t_base = None, tauo = None, channel=0, px_wise = False):
	# return empirical tau series for all steps in the image
	flimobj = FlimObj.FlimObj()
	flimobj.read_FLIM(file_path)
	flimobj.times_to_datetime() # adds an attribute to the flimobj that stores the times as datetimes (flimobj.datetimes)
	imarray = flimobj.get_images_as_intensity_array(flimbins=True)
	if tauo:
		x_in_seconds, emp_taus = flimobj.fit_tau(imarray, avg=avg,tauo=tauo,channel=channel)
	else:
		x_in_seconds, emp_taus = flimobj.fit_tau(imarray,avg=avg, channel=channel)
	if px_wise:
		flimobj.flimtiff(flim=True) # save a .tiff file with the taus for each pixel
	if t_base:
		t_offset = (flimobj.datetimes[0]-t_base).total_seconds()
		x_in_seconds = x_in_seconds+t_offset
	return x_in_seconds, emp_taus, flimobj

def pulsed_t_series(t_axis, fluor, flimobj,avg=1):
	# handle the pulses by turning them into nans, return the times of pulses
	pulse_params = flimobj.FLIMageFileReader.State.Uncaging
	# params (in milliseconds)
	(nPulses, pulsewidth, isi, delay) = (pulse_params.nPulses, pulse_params.pulseWidth, pulse_params.pulseISI, pulse_params.pulseDelay)
	pidx_list = []
	p_offset = pulse_params.baselineBeforeTrain_forFrame
	for p in range(nPulses):
		pstart = delay + p*(isi)+p_offset
		pend = p*(isi) +delay + pulsewidth+p_offset # in milliseconds
		# now measure from start to end of each pulse
		pidxs = np.floor(np.arange(flimobj.last_frame_before_time(pstart, units="milliseconds")-1,\
			flimobj.last_frame_before_time(pend, units="milliseconds"),avg)/avg).astype(int).tolist()
		fluor[pidxs] = np.nan
		pidx_list.append(pidxs)
	return fluor, pidx_list

def single_round_process(file_path,plot=False,tau_offset=None, avg=1, t_base = None, channel = 0, px_wise = False):
	# everything I want to do to a single trace
	t_axis, emp_taus, flimobj = flim_analysis(file_path, avg=avg, t_base = t_base, tauo=tau_offset, channel=channel, px_wise = px_wise) # the flimobj contains a lot of additional info
	pidxs=[[0]]
	if flimobj.FLIMageFileReader.State.Uncaging.uncage_whileImage:
		emp_taus, pidxs = pulsed_t_series(t_axis, emp_taus, flimobj,avg=avg) # deal with the pulses, find out which frames correspond to them
	if not t_base:
		t_axis += -t_axis[pidxs[0][0]] # subtract the start of the pulse
	# now write t and dfof to an excel file
	writer = pd.ExcelWriter('%s_flim_%s.xlsx'%(file_path[:-5],channel), engine='xlsxwriter')
	pd.DataFrame(np.vstack((t_axis, emp_taus)).T).to_excel(writer)
	writer.save()

	if plot:
		plt.plot(t_axis,dfof)
		plt.show()
	return t_axis, emp_taus, flimobj

def main(argv):
	# extract number of frames to average
	unixOpts = "a:c:sp"
	gnuOpts = ["average=","channels=","save=","pixel_wise="]
	try:
		avg=1
		channels=1
		save=False
		px = False
		arguments, values = getopt.getopt(argv, unixOpts, gnuOpts)
		for opt, arg in arguments:
			print(opt,arg)
			if opt in ('-a',"--average"):
				avg = int(arg)
			elif opt in ('-s', '--save'):
				save = True
			elif opt in ('-c', '--channels'):
				channels = int(arg)
			elif opt in ('-p', '--pixel_wise'):
				px = True
	except getopt.GetoptError as err:
		print(str(err))
		usage()
		sys.exit(2)

	# find the file, extract the base name
	root = Tk.Tk()
	root.withdraw()
	root.update()
	#direc = filedialog.askdirectory()
	file_path = filedialog.askopenfilename()
	root.update()

	basename = re.match("(.*\D)[0-9]+\.flim",file_path).group(1)  # up to the part that ends in <numbers>.flim
	filelist = sorted(glob.glob("%s*.flim"%(basename)),key=lambda x: int(re.findall("(\d+)",x)[-1])) # sort numerically

	first_val = [True for c in range(channels)]
	t_list = [[] for c in range(channels)]
	tau_list = [[] for c in range(channels)]
	tauo_list = [None for c in range(channels)]
	t_base = [None for c in range(channels)]
	for file_path in filelist:
		for c in range(channels):
			t_vals, emp_taus, flimobj = single_round_process(file_path, t_base = t_base[c], tau_offset=tauo_list[c], avg=avg, channel = c, px_wise=px)
			if first_val[c]:
				t_base[c] = flimobj.datetimes[0]
				tauo_list[c]=  flimobj.tauo
				first_val[c] = False
			t_list[c].append(t_vals)
			tau_list[c].append(emp_taus)

	t_axis = [np.concatenate(t_list[c], axis = 0).flatten() for c in range(channels)]
	taus = [np.concatenate(tau_list[c], axis = 0).flatten() for c in range(channels)]
	t_axis = t_axis[0]
	taus = np.array(taus)

	if save:
		concat_trace = pd.DataFrame(np.vstack((t_axis.T,taus)).T)
		sum_writer = pd.ExcelWriter('%s--stitched_flim.xlsx' %(basename),engine='xlsxwriter')
		concat_trace.to_excel(sum_writer)
		sum_writer.save()
	for c in range(channels):
		plt.plot(t_axis,taus[c,:])
	plt.show()

if __name__ == "__main__":
	main(sys.argv[1:])
	

