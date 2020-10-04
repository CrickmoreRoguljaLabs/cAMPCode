import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
	import tkinter as tk
import matplotlib as mpl
import matplotlib.pyplot as plt
import FLIMageFileReader
import numpy as np
from tkinter import filedialog
import re, glob
import datetime


def lfplot(lifetime_map,bounds=[1.4,1.9]):
	# plot a colored map of pixelwise fluorescence lifetime
	lower_bound, upper_bound = bounds[0], bounds[1]
	plt.figure()
	plt.imshow(zd_lf)
	plt.clim(lower_bound,upper_bound)
	plt.colorbar()
	plt.show()

if __name__ == "__main__":
	#plotWindow = tk.Tk()
	#plotWindow.wm_title('Fluorescence lifetime')                
	#plotWindow.withdraw()
	
	# find the file, extract the base name
	file_path = filedialog.askopenfilename()
	basename = re.match("(.*\D)[0-9]+\.flim",file_path).group(1)  # up to the part that ends in <numbers>.flim
	filelist = sorted(glob.glob("%s*.flim"%(basename)),key=lambda x:float(re.findall("(\d+)",x)[0])) # sort numerically
	num_files = len(filelist)
	emp_tau = np.zeros((num_files,))
	t_axis = np.zeros((num_files,))
	i = 0
	for file in filelist:
		iminfo = FLIMageFileReader.FileReader()

		iminfo.read_imageFile(file, True)
		iminfo.calculatePage(0, 0, 0, [0, iminfo.n_time[0]], [0, 10], [1.6, 3], 1.5)
		if i == 0 :
			start_time = datetime.datetime.strptime(iminfo.acqTime[0],"%Y-%m-%dT%H:%M:%S.%f")
		this_time = time = datetime.datetime.strptime(iminfo.acqTime[0],"%Y-%m-%dT%H:%M:%S.%f")
		diff = this_time - start_time
		hist_x = iminfo.time
		hist_y = iminfo.lifetime
		lifetime_offset=1.5 # just playing for now
		#t_axis[i] = iminfo.acqTime
		#['2019-04-16T12:27:29.155']
		t_axis[i] = diff.total_seconds()/60.0
		emp_tau[i] = np.sum(hist_x*hist_y)/np.sum(hist_y) - lifetime_offset
		zd_lf = iminfo.lifetimeMap
		thresh=10.0
		zd_lf[iminfo.intensity < thresh] = 0
		i += 1
		#lfplot(zd_lf)
plt.plot(t_axis,emp_tau)
plt.show()
