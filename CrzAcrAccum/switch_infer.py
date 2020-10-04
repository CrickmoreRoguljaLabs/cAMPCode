import pystan
import numpy as np
import pandas as pd

sm = pystan.StanModel(file='switch.stan') # define the model using the "switch.stan" file

### TODO: crz_data
df_dict = pd.read_excel("Window Accumulation Experiments Charlotte.xlsx", sheet_name = None)

# the data for the "control" condition
control_sheets = ['CS x DCA light off at 10 min']
off_data = df_dict['CS x DCA light off at 10 min']

lightoff = off_data['green light off at (min)'].values
termoff = off_data['mating duration (min)'].values

# data for the experimental condition

# Haven't come up with an elegant way to do all the inference together (which seems to make this whole exercise seem a little silly)
posterior_fits = {} # dict for each set of posteriors
summary = {}
for (sheet_name,  df) in df_dict.items():
	if not sheet_name in control_sheets: # skip the control sheet
		posterior_fits[sheet_name] = {}
		summary[sheet_name] = {}
		sheet_name = df["Duration of window per cycle (sec)"][0]
		each_cond = df.groupby(["Duration of green light per cycle (sec)"])
		for light_length, data in each_cond:
			print("Window: %s, Pulse: %s" %(sheet_name, light_length))
			termaccum = data['mating duration (min)'].convert_objects(convert_numeric=True).dropna().values
			print(termaccum)
			# data to feed into the stan model
			crz_data = {'Noff': len(termoff),
				'termoff': termoff,
				'lightoff': lightoff,
				'Naccum': len(termaccum),
				'termaccum': termaccum
			}
			fit = sm.sampling(data=crz_data, iter=4000, chains=4, warmup=1000, thin=3) # fit the model using the data defined above
			posterior_fits[sheet_name][light_length] = fit.extract(permuted=True) 
			summary[sheet_name][light_length] = fit.summary()
np.save('fits.npy', posterior_fits)
np.save('summary.npy', summary)

