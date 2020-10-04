import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 3
mpl.rcParams['ps.fonttype'] = 3
plt.rcParams['axes.linewidth']=0.5
plt.rcParams['xtick.major.width']=0.5
plt.rcParams['ytick.major.width']=0.5
labelsize=9
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
plt.rcParams['legend.fontsize'] = labelsize

hfont = {'fontsize':labelsize,'fontname':'Arial'}

plt.rcParams.update()

ticks_font = mpl.font_manager.FontProperties(family='Arial', style='normal', size=labelsize, weight='normal', stretch='normal')

# color dictionary for each pulse width

cdict = {
10:[0.9*0.8/0.6,.7*0.8/0.6,.0],
20:[.91*0.8/0.6,.58*0.8/0.6,.02],
30:[.84*0.8/0.6,.91*0.8/0.6,.04],
40:[.47,.92,.06],
50:[.11,.93,0.08],
60:[.10,.94,0.45],
70:[0.13,.95,.83],
80:[0.15,.71,.95],
90:[0.17,.37,.96],
100:[0.34,.19,.97],
110:[0.70,.22,.98]
}

bright = 0.6

for (pw, col) in cdict.items():
	cdict[pw] = np.array(col)*bright

# histogram of the posterior distribution on the switch time

posterior_fits = np.load('fits.npy').item()

# first_el = posterior_fits[15][10]
# plt.figure(figsize=(1.6,1.2))
# plt.hist(first_el['mu'], alpha=0.5, color='red')
# plt.xlabel('Mean of Gaussian (min)')
# plt.ylim((0,2000))
# plt.savefig("mu_dist.pdf", dpi=600, facecolor=None, edgecolor=None,
#         orientation='portrait', papertype=None, format='pdf',
#         transparent=True, bbox_inches=None, pad_inches=0.1,
#         frameon=None, metadata=None)
# plt.figure(figsize=(1.6,1.2))
# plt.hist(first_el['sig'], alpha=0.5, color='blue')
# plt.xlabel('Std of Gaussian (min)')
# plt.ylim((0,2000))
# plt.savefig("sig_dist.pdf", dpi=600, facecolor=None, edgecolor=None,
#         orientation='portrait', papertype=None, format='pdf',
#         transparent=True, bbox_inches=None, pad_inches=0.1,
#         frameon=None, metadata=None)
# plt.show()

for (window_width, dic) in posterior_fits.items():
	plt.figure(figsize=(1.6,1.2))
	plt.title('%s' %(window_width), **hfont)
	for (pulse_width, data) in dic.items():
		for (param, trace) in data.items():
			if param == "tauaccum":
				plt.hist(trace, alpha=0.5, color=cdict[int(pulse_width)])
	plt.legend()
	plt.xlabel('Inferred time of switch (min)', **hfont)
	plt.ylabel('Probability density', **hfont)
	plt.xlim((0,180))
	plt.ylim((0,2000))
	plt.xticks(np.arange(0.0,181,60))
	plt.yticks(np.arange(0.0,2001,1000))
	plt.savefig("%s_window.pdf" %(window_width,), dpi=600, facecolor=None, edgecolor=None,
        orientation='portrait', papertype=None, format='pdf',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

# histogram of the posterior distribution of "cumulative time spent with no light"
for (window_width, dic) in posterior_fits.items():
	plt.figure(figsize=(1.5,1.2))
	plt.title('%s' %(window_width), **hfont)
	for (pulse_width, data) in dic.items():
		for (param, trace) in data.items():
			frac_light_off = float(window_width)/((float(window_width) + float(pulse_width)))
			on_time = float(pulse_width) / float(window_width)
			if param == "tauaccum":
				trans = (trace-11.0)*(frac_light_off)
				plt.hist(trans, alpha=0.5, color=cdict[int(pulse_width)])
				print("Window width: %s pulse width: %s xmean:%s xsem: %s ymean: %s ysem: %s" %(window_width, pulse_width, np.mean((trans+1)*on_time), np.std((trans+1)*on_time), np.mean((trans+1)),np.std((trans+1))))
	plt.legend()
	plt.xlabel('Inferred cumulative duration of\nactivity to cause switch (min)', **hfont)
	plt.ylabel('Probability density', **hfont)
	plt.xlim((0,40))
	plt.ylim((0,2000))
	plt.xticks(np.arange(0,41,20))
	plt.yticks(np.arange(0.0,2001,1000))
	plt.savefig("%s_cumulative_time.pdf" %(window_width,), dpi=600, facecolor=None, edgecolor=None,
        orientation='portrait', papertype=None, format='pdf',
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)



#plt.show()