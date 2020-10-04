# for batch processing of t series that are similar
import FlimObj
from tkinter import filedialog
import re, glob, os
import matplotlib.pyplot as plt
import datetime
import numpy as np
import tkinter as Tk
import pandas as pd


def dfof_analysis(file_path):
	# compute dfof using the sum of fluorescence in an image (no ROIs in this analysis)
	flimobj = FlimObj.FlimObj()
	flimobj.read_FLIM(file_path)
	flimobj.times_to_datetime() # adds an attribute to the flimobj that stores the times as datetimes (flimobj.datetimes)
	image_array = flimobj.get_images_as_intensity_array()
	x_in_seconds = flimobj.get_t_axis() # it's in seconds

	pulse_t = flimobj.get_pulse_delay()
	pulse_frame = np.sum(x_in_seconds<pulse_t) + flimobj.get_pulse_frames_baseline()

	summed_vals = np.sum(image_array-np.median(image_array),(3,4))
	F_base = np.mean(summed_vals[(pulse_frame-5):(pulse_frame-1),0,0])
	dfof = (summed_vals[:,0,0]-F_base)/F_base
	return x_in_seconds-pulse_t, dfof, flimobj

def pulsed_t_series(t_axis, dfof, flimobj):
	# handle the pulses by turning them into nans, return the times of pulses
	pulse_params = flimobj.FLIMageFileReader.State.Uncaging
	# params (in milliseconds)
	(nPulses, pulsewidth, isi, delay) = (pulse_params.nPulses, pulse_params.pulseWidth, pulse_params.pulseISI, pulse_params.pulseDelay)
	pidx_list = []
	p_offset = pulse_params.baselineBeforeTrain_forFrame
	for p in range(nPulses):
		pstart = delay + p*(isi) +p_offset
		pend = p*(isi+pulsewidth)+delay + pulsewidth + p_offset # in milliseconds
		# now measure from start to end of each pulse
		pidxs = range(flimobj.last_frame_before_time(pstart, units="milliseconds")-1,\
			flimobj.last_frame_before_time(pend, units="milliseconds"))
		dfof[pidxs] = np.nan
		pidx_list.append(pidxs)
	return dfof, pidx_list

def return_max_and_sustained(dfof):
	# iterate through each pulse, pull out the max dfof for each and the sustained dfofs
	maxlist = []
	suslist = []
	datatrace = np.ravel(np.argwhere(1.0-np.isnan(dfof))) # idxs of frames with data
	jump_idxs = datatrace[np.pad((datatrace[1:-1]-datatrace[0:-2]) > 1,1,'constant',constant_values=(0,0))] # it's a pulse if there's at least one frame in between.
	jump_idxs = np.append(jump_idxs,datatrace[-1])
	# now iterate through the jumps and find the max and sus
	for pulse_id in range(len(jump_idxs)-1):
		pulse_frames = dfof[jump_idxs[pulse_id]:jump_idxs[pulse_id+1]] # everything between this and the next pulse
		maxlist.append(np.nanmax(pulse_frames))
		try:
			suslist.append(np.nanmean(pulse_frames[60:80]))
		except:
			suslist.append(np.nan())
	return maxlist, suslist

def single_round_process(dirpath,filename,plot=False):
	# everything I want to do to a single trace
	file_path = os.path.join(dirpath, filename)
	t_axis, dfof, flimobj = dfof_analysis(file_path) # the flimobj contains a lot of additional info
	dfof, pidxs = pulsed_t_series(t_axis, dfof, flimobj) # deal with the pulses, find out which frames correspond to them
	maxs, sus = return_max_and_sustained(dfof)

	# now write t and dfof to an excel file
	writer = pd.ExcelWriter('%s/%s.xlsx'%(dirpath,filename[:-5]), engine='xlsxwriter')
	pd.DataFrame(np.vstack((t_axis, dfof)).T).to_excel(writer)
	writer.save()

	if plot:
		plt.plot(t_axis,dfof)
	return maxs, sus

if __name__ == "__main__":
	
	# find the file, extract the base name
	root = Tk.Tk()
	root.withdraw()
	root.update()
	direc = filedialog.askdirectory()
	root.update()

	save = True
	if save:
		summary_values = pd.DataFrame()
		for dirpath, dirnames, filenames in os.walk(direc):
			for filename in [f for f in filenames if f.endswith(".flim")]:
				plot = True
				maxs, sus = single_round_process(dirpath,filename,plot=plot)
				ppr = np.nan
				sus_rat = np.nan
				if len(maxs) > 1:
					ppr = maxs[1]/maxs[0]
					sus_rat = sus[1]/sus[0]
				else:
					maxs = [maxs,maxs]
					sus = [sus,sus]
				if maxs[0]:
					summary_values = pd.concat([summary_values,pd.DataFrame(data={"File":"%s/%s"%(dirpath,filename),\
						"Max":maxs[0],"Max2":maxs[1],"Sustained":sus[0], "Sustained2":sus[1], \
						"PPR":ppr, "Sustained Ratio":sus_rat},index=[0])])
		sum_writer = pd.ExcelWriter('%s/Summary.xlsx'%(direc),engine='xlsxwriter')
		summary_values.to_excel(sum_writer)
		sum_writer.save()
		if plot:
			plt.show()

	#file_path = filedialog.askopenfilename()

	#basename = re.match("(.*\D)[0-9]+\.flim",file_path).group(1)  # up to the part that ends in <numbers>.flim
	#filelist = sorted(glob.glob("%s*.flim"%(basename)),key=lambda x:float(re.findall("(\d+)",x)[0])) # sort numerically

	#single_round_process(file_path) # process each trace



