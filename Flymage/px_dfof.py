# for batch processing of t series that are similar
import FlimObj
from tkinter import filedialog
import re, glob, os
import matplotlib.pyplot as plt
import datetime
import numpy as np
import scipy.signal
import tkinter as Tk
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def dfof_analysis(file_path):
	# compute dfof using the sum of fluorescence in an image (no ROIs in this analysis)
	flimobj = FlimObj.FlimObj()
	flimobj.read_FLIM(file_path)
	flimobj.times_to_datetime() # adds an attribute to the flimobj that stores the times as datetimes (flimobj.datetimes)
	image_array = flimobj.get_images_as_intensity_array()
	x_in_seconds = flimobj.get_t_axis() # it's in seconds

	pulse_t = flimobj.get_pulse_delay()
	pulse_frame = np.sum(x_in_seconds<pulse_t)
	# median filter first
	image_array = scipy.signal.medfilt(image_array,kernel_size=[1,1,1,3,3])
	px_vals = image_array-np.median(image_array)
	F_base_px = np.mean(px_vals[(pulse_frame-5):(pulse_frame-1),0,0])
	dfof = (px_vals[:,0,0]-F_base_px)/F_base_px
	return x_in_seconds-pulse_t, dfof, flimobj, px_vals

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
		plt.show()
	return maxs, sus

if __name__ == "__main__":
	
	# find the file, extract the base name
	root = Tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename()
	t_axis, dfof, flimobj, im_data = dfof_analysis(file_path)
	raw_ims = im_data[:,0,0,:,:]
	print(np.max(raw_ims))
	RdBu=cm.get_cmap("viridis")
	colormap=RdBu(np.linspace(0,1.0,256))
	#colormap=colormap[::-1]
	colormap[0]=np.array([0,0,0,1])
	print(raw_ims.shape)
	colormap=ListedColormap(colormap)
	colormap.set_bad('black')

	vmin = 0.0
	vmax = 0.7

	ax= plt.imshow(raw_ims[34,:,:]/9.0,cmap=colormap,vmin=vmin,vmax=vmax)
	plt.gca().get_xaxis().set_visible(False)
	plt.gca().get_yaxis().set_visible(False)
	plt.figure()
	ax = plt.imshow(raw_ims[79,:,:]/9.0,cmap=colormap,vmin=vmin,vmax=vmax)
	plt.gca().get_xaxis().set_visible(False)
	plt.gca().get_yaxis().set_visible(False)
	plt.figure()
	ax =plt.imshow(raw_ims[129,:,:]/9.0,cmap=colormap,vmin=vmin,vmax=vmax)
	plt.gca().get_xaxis().set_visible(False)
	plt.gca().get_yaxis().set_visible(False)
	plt.figure()
	plt.imshow(raw_ims[129,:,:]/9.0,cmap=colormap,vmin=vmin,vmax=vmax)
	plt.colorbar()
	plt.show()



	#basename = re.match("(.*\D)[0-9]+\.flim",file_path).group(1)  # up to the part that ends in <numbers>.flim
	#filelist = sorted(glob.glob("%s*.flim"%(basename)),key=lambda x:float(re.findall("(\d+)",x)[0])) # sort numerically

	#single_round_process(file_path) # process each trace



