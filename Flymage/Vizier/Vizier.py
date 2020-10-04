import numpy as np
from Vizier.imageScroller import ImScroll
import tkinter as tk
# general image-related information to use for plotting in fancy ways.

## A vizier keeps track of several linked figures and plots, and ties them together with a scroll. (well, that's intended at least)
## The scroll synchronizes the display of several figures that are linked and trying to show related information

class Vizier(object):
	def __init__(self):
		self.cmap_dict = {}
		self.axes_dict = {}
		self.master = tk.Tk() # master window

	def image_from_array(self,image_array,axes_dict = {'z':0,'x':1,'y':2,'c':3,'t':4}):
		# axes_dict explains which axis should correspond to a z plane, t point, x axis, y axis, or arbitrary axes name
		# c is for color, t is for time
		self.image_arrays = np.transpose(image_array,axes=(axes_dict['z'],axes_dict['x'],axes_dict['y'],axes_dict['c'],axes_dict['t'])) # remap the axes
		self.axes_dict = axes_dict

	def show_image(self, colormap=None, color_channel=0,**kwargs):
		scroller = ImScroll(self.master,**kwargs)
		scroller.display(self.image_arrays, color=color_channel)
		self.scroller = scroller
		return scroller