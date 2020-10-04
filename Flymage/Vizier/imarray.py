## An Imarray keeps track of a figure plotted as an RGB map + single channel images to the side.
## SCT

import numpy as np
import matplotlib.pyplot as plt

class Imarray(object):
	# a class that makes it a little easier to work with images within a larger set of figures

	def __init__(self, image_arrays, num_colors = 3, colormap=None):
		# must be initialized with a pyplot axes and an image arrays that will scroll in the first dimension
		self.figure = plt.figure()
		merge_ax = plt.subplot2grid((num_colors, 4), (0, 0), rowspan=num_colors, colspan=3)
		self.image_arrays = image_arrays
		self.type = "image_stack"
		self.curr_pos = 0
		self.color_axes = {}

		num_colors = image_arrays.shape[-1]
		# create the figure itself
		self.axes = plt.imshow(self.image_arrays[self.curr_pos,:,:,:], cmap=colormap)
		self.main_axes = plt.gca()
		merge_ax.get_xaxis().set_visible(False)
		merge_ax.get_yaxis().set_visible(False)

		# single color channel
		if num_colors > 3:
			self.lots_of_colors = True
		else:
			self.lots_of_colors = False
		if self.lots_of_colors: # if the colormap is > 3, make a bunch of subplot channels
			for k in range(num_colors):
				ax = plt.subplot2grid((num_colors,1),(k,3))
				if colormap:
					self.color_axes[k]=plt.imshow(self.image_arrays[self.curr_pos,:,:,k],cmap=colormap)
					plt.colorbar()
				else:
					self.color_axes[k]=plt.imshow(self.image_arrays[self.curr_pos,:,:,k],cmap="gray")
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
		else:
			for k in range(num_colors):
				ax = plt.subplot2grid((3,4),(k,3))
				if colormap:
					self.color_axes[k]=plt.imshow(self.image_arrays[self.curr_pos,:,:,k],cmap=colormap)
					plt.colorbar()
				else:
					self.color_axes[k]=plt.imshow(self.image_arrays[self.curr_pos,:,:,k],cmap="gray")
				ax.set_title("Channel %s" %(k+1))
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)

	def update_axis(self, update_val):
		# what to do when the scroller updates
		self.axes.set_array(self.image_arrays[update_val,:,:,:])
		for color,ax in self.color_axes.items():
					ax.set_array(self.image_arrays[self.curr_pos,:,:,color])
		self.curr_pos = update_val