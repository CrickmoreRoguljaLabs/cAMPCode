import numpy as np
import pandas as pd


df_dict = pd.read_excel("Window Accumulation Experiments Charlotte.xlsx", sheetname = None)

# the data for the "control" condition
control_sheets = ['CS x DCA light off at 10 min']
off_data = df_dict['CS x DCA light off at 10 min']

lightoff = off_data['green light off at (min)'].values
termoff = off_data['mating duration (min)'].values

# data for the experimental condition

# Haven't come up with an elegant way to do all the inference together (which seems to make this whole exercise seem a little silly)
posterior_fits = {} # dict for each set of posteriors
for (sheet_name,  df) in df_dict.items():
	if not sheet_name in control_sheets: # skip the control sheet
		posterior_fits[sheet_name] = {}
		each_cond = df.groupby(["Duration of green light per cycle (sec)"])
		for light_length, data in each_cond:
			print("Window: %s, Pulse: %s" %(sheet_name, light_length))
			termaccum = data['mating duration (min)'].dropna().values
			print(termaccum.dtype)
			# data to feed into the stan model
			crz_data = {'Noff': le.valuesn(termoff),
				'termoff': termoff,
				'lightoff': lightoff,
				'Naccum': len(termaccum),
				'termaccum': termaccum
			}
