# converts a .flim file into a .tiff one
import FlimObj
from tkinter import filedialog
import re, glob, os
import datetime
import numpy as np
import tkinter as Tk
import sys, getopt
import tifffile as tiff

def single_round_process(filename,median_filter=1, use_FLIM = False):
	# everything I want to do to a single trace
	flimobj = FlimObj.FlimObj()
	flimobj.read_FLIM(filename)
	image_array = flimobj.get_images_as_intensity_array(flimbins=use_FLIM) # returns intensity array
	image_array = image_array.astype(np.uint32)
	print(image_array.shape)
	tiff.imsave("%s.tiff"%(os.path.splitext(filename)[0]),image_array)


def main(argv):

	unixOpts = "f"
	gnuOpts = ["filt="]
	try:
		filt = 1
		arguments, values = getopt.getopt(argv, unixOpts, gnuOpts)
		for opt, arg in arguments:
			print(opt,arg)
			if opt in ('-f',"--filt"):
				filt = int(arg)
	except getopt.GetoptError as err:
		print(str(err))
		usage()
		sys.exit(2)

	# extract number of frames to average
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

	for file in filelist:
		single_round_process(file, median_filter=filt)

	root.destroy()

if __name__ == "__main__":
	main(sys.argv[1:])