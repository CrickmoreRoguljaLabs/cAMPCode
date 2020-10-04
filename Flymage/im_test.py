import FlimObj
from tkinter import filedialog
import re
import matplotlib.pyplot as plt

if __name__ == "__main__":
	#plotWindow = tk.Tk()
	#plotWindow.wm_title('Fluorescence lifetime')                
	#plotWindow.withdraw()
	
	# find the file, extract the base name
	file_path = filedialog.askopenfilename()
	basename = re.match("(.*\D)[0-9]+\.flim",file_path).group(1)  # up to the part that ends in <numbers>.flim
	flimobj = FlimObj.FlimObj()
	t_stack, t_axis = flimobj.read_t_stack(basename, file_path)
	flimobj.viz_t_stack()
	plt.show()