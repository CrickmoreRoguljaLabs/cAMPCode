## A class to allow scrolling through z-stack images
## SCT

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from PIL import Image, ImageTk

class ImScroll(object):
    """ For scrolling through images """ 
    def __init__(self, master, crange=(0.0,255.0)):
        self.figures = [] # list of all figures linked to this scroller
        self.master = master # master window
        self.c_max = crange[1]
        self.c_min = crange[0]

    def display(self, image_arrays, color=0):
        ### set up the main display
        self.image_array = image_arrays
        self.color = color
        self.frame_dims = [self.image_array.shape[1],self.image_array.shape[2]]
        self.main_frame = tk.Frame(self.master)
        self.master.title("Vizier")
        #self.main_frame.pack_propagate(0)
        self.curr_z_pos = 0
        self.curr_t_pos = 0
        
        # select time point
        self.t_size = self.image_array.shape[-1]
        self.t_scroll_frame = tk.Frame(self.main_frame, height = 3) # split off the scroll bars to put in info
        self.t_scroll_frame.pack(side=tk.BOTTOM, anchor=tk.N)
        self.t_scroll = tk.Scale(self.t_scroll_frame, orient=tk.HORIZONTAL, from_=1, to_=self.t_size, label="Frame number")
        self.t_scroll.pack(anchor=tk.N, side=tk.BOTTOM)
        self.t_scroll.config(command=lambda x: self.update_t())
        
        # info for z slice
        self.z_size = self.image_array.shape[0]
        self.z_scroll_frame = tk.Frame(self.main_frame, width=10) # split off the scroll bars to put in info
        self.z_scroll_frame.pack(side=tk.RIGHT)
        self.z_scroll = tk.Scale(self.z_scroll_frame, orient=tk.VERTICAL,from_=1, to_=self.z_size, label="Z slice")
        self.z_scroll.pack(side=tk.RIGHT)
        self.z_scroll.config(command=lambda x: self.update_z())
        
        # The actual image
        #self.image_canvas = tk.Canvas(self.main_frame, width=self.frame_dims[0], height=self.frame_dims[1])
        self.image_frame = tk.Frame(self.master, width=self.frame_dims[0], height=self.frame_dims[1])
        self.image_frame.pack(side=tk.LEFT,anchor=tk.NW, fill=tk.BOTH, expand=1)
        self.image_canvas = tk.Canvas(self.image_frame)
        self.image_canvas.pack(fill=tk.BOTH,expand=1)
        self.image = self.image_obj(image_arrays[self.curr_z_pos,:,:,color,self.curr_t_pos])
        self.image_on_canvas = self.image_canvas.create_image(0,0,anchor=tk.NW,image=self.image)

        # Select an ROI
        self.button_frame = tk.Frame(self.master)
        self.roi_button = tk.Button(self.button_frame, text="Select ROI")
        self.roi_button.config(command = lambda: self.select_roi())
        self.done_button = tk.Button(self.button_frame, text="Done")
        self.done_button.config(command= lambda: self.close())
        self.roi_button.pack(side=tk.LEFT)
        self.done_button.pack(side=tk.RIGHT)
        self.button_frame.pack(side=tk.BOTTOM)
        self.roi_list = [[] for z in range(self.image_array.shape[0])]
        self.roi_coords = [[] for z in range(self.image_array.shape[0])]
        #
        self.main_frame.pack()
        self.master.mainloop()

    def image_obj(self, array):
        ### take an np.array object and draw it on the image_canvas
        datarray = np.squeeze(
            np.maximum(
                (255.0/(self.c_max-self.c_min))*(array-self.c_min), 0.0
                ) 
            )
        im = ImageTk.PhotoImage(master=self.image_canvas,image=Image.fromarray(datarray.astype('uint8')))
        return im

    def update_t(self):
        self.curr_t_pos = self.t_scroll.get()-1
        self.image = self.image_obj(self.image_array[self.curr_z_pos,:,:,self.color,self.curr_t_pos])
        self.image_canvas.itemconfig(self.image_on_canvas, image=self.image)

    def update_z(self):
        # hide roi from other slices
        for roi in self.roi_list[self.curr_z_pos]:
            self.image_canvas.itemconfig(roi, state='hidden')
        
        self.curr_z_pos = self.z_scroll.get()-1
        self.image = self.image_obj(self.image_array[self.curr_z_pos,:,:,self.color,self.curr_t_pos])
        self.image_canvas.itemconfig(self.image_on_canvas, image=self.image)
        
        # show previously-made rois
        for roi in self.roi_list[self.curr_z_pos]:
            self.image_canvas.itemconfig(roi, state='normal')

    def update_crange(self,minimax):
        # updates the colormap min and max values
        self.c_min=minimax[0]
        self.c_max=minimax[1]

    def select_roi(self):
        self.image_canvas.bind("<Button-1>",lambda event: self.roi_init(event))

    def roi_init(self, event):
        # start bounding the ROI
        print("Clicked!")
        roi_first_coords = (event.x,event.y)
        self.roi_list[self.curr_z_pos].append(self.image_canvas.create_rectangle(event.x,event.y,event.x,event.y,outline='blue', width=2))
        self.image_canvas.bind("<B1-Motion>", lambda event: self.roi_draw(event,roi_first_coords))
        self.image_canvas.bind("<ButtonRelease-1>", lambda event: self.roi_end(event, roi_first_coords))

    def roi_draw(self, event, coordinates):
        # stretch the rectangle to match where the mouse is
        self.image_canvas.coords(self.roi_list[self.curr_z_pos][-1],coordinates[0],coordinates[1],event.x,event.y)

    def roi_end(self,event, coords):
        # save the box of the roi
        roi_end_coords = (event.x, event.y)
        self.roi_coords[self.curr_z_pos].append((coords,roi_end_coords))
        self.image_canvas.unbind("<Button-1>")
        self.image_canvas.unbind("<B1-Motion>")
        self.image_canvas.unbind("<ButtonReleases-1>")

    def close(self):
        #print(self.roi_coords)
        self.master.withdraw()
        self.master.quit()
        self.master.destroy()
