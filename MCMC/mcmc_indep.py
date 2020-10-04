### use the model from indep_samples.py

import numpy as np
import matplotlib.pyplot as plt
import pymc
import indep_samples
import Tkinter as tk

def run_mcmc(x1,f1,x2,f2):
	S = pymc.MCMC(indep_samples.indep_samples(x1,x1+f1,x2,f2), db='pickle')
	S.sample(iter=100000, burn=10000, thin=3)

	print '\n'
	print 'MEAN POSTERIOR ESTIMATES: \n' + 'p1: '+str(S.p1.stats()['mean']) + '\n' + \
			 'p2: ' + str(S.p2.stats()['mean']) + '\n' +\
			  'independence: ' + str(S.ind_assump.stats()['mean']) + '\n'

	print '\n'

	p1_values = S.p1.trace()
	p2_values = S.p2.trace()
	#sim_data_vals = S.sim.trace()
	ind_values = S.ind_assump.trace()

	print "68% confidence intervals: \n"
	print 'p1: ' + str([np.percentile(p1_values,16),np.percentile(p1_values,84)])
	print 'p2: ' + str([np.percentile(p2_values,16),np.percentile(p2_values,84)])
	#print 'sim sample: ' + str([np.percentile(sim_data_vals,16),np.percentile(sim_data_vals,84)])
	print 'independence: ' + str([np.percentile(ind_values,16),np.percentile(ind_values,84)])

	print '\n'

	print "95% confidence intervals: \n"
	print 'p1: ' + str([np.percentile(p1_values,2.5),np.percentile(p1_values,97.5)])
	print 'p2: ' + str([np.percentile(p2_values,2.5),np.percentile(p2_values,97.5)])
	#print 'sim sample: ' + str([np.percentile(sim_data_vals,2.5),np.percentile(sim_data_vals,97.5)])
	print 'independence: ' + str([np.percentile(ind_values,2.5),np.percentile(ind_values,97.5)])

	observed = .564

	print 'p value: '+str((ind_values > observed).sum()/float(ind_values.shape[0]))

	return [[float(x1)/(x1+f1), np.percentile(p1_values,16) , np.percentile(p1_values,84)],...
		[float(x2)/(x2+f2), np.percentile(p2_values,16) , np.percentile(p2_values,84)], ...
		[float(x1)/(x1+f1) + float(x2)/(x2+f2) - float(x1*x2)/((x1+f1)*(x2+f2), np.percentile(ind_values,16),np.percentile(ind_values,84)]]

	#pymc.Matplot.plot(S)

master = tk.Tk()

def init_inf():
   var1, var2, ind = run_mcmc(int(e1.get()),int(e2.get()))
   samp_entry.insert(0,str(var1[0]))
   upper_entry.insert(0,str(var1[1]))
   lower_entry.insert(0,str(var1[2]))

   samp_entry2.insert(0,str(var2[0]))
   upper_entry2.insert(0,str(var2[1]))
   lower_entry2.insert(0,str(var2[2]))

   ind_prop.insert(0,str(ind[0]))
   ind_up.insert(0,str(ind[1]))
   ind_low.insert(0,str(ind[2]))


tk.Label(master, text="Number of successes").grid(row=0)
tk.Label(master, text="Number of failures").grid(row=1)

tk.Label(master, text="Sample proportion").grid(row=0, column = 2)
tk.Label(master, text="Upper interval window").grid(row=1, column = 2)
tk.Label(master, text="Lower interval window").grid(row=2, column = 2)

tk.Label(master, text="Number of successes").grid(row=0, column = 4)
tk.Label(master, text="Number of failures").grid(row=1, column = 4)

tk.Label(master, text="Sample proportion").grid(row=0, column = 6)
tk.Label(master, text="Upper interval window").grid(row=1, column = 6)
tk.Label(master, text="Lower interval window").grid(row=2, column = 6)

tk.Label(master, text="Indep proportion").grid(row=0, column = 8)
tk.Label(master, text="Upper interval window").grid(row=1, column = 8)
tk.Label(master, text="Lower interval window").grid(row=2, column = 8)

e1 = tk.Entry(master)
e2 = tk.Entry(master)
samp_entry = tk.Entry(master)
upper_entry = tk.Entry(master)
lower_entry = tk.Entry(master)

e3 = tk.Entry(master)
e4 = tk.Entry(master)
samp_entry2 = tk.Entry(master)
upper_entry2 = tk.Entry(master)
lower_entry2 = tk.Entry(master)

ind_prop = tk.Entry(master)
ind_up = tk.Entry(master)
ind_low = tk.Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
samp_entry.grid(row=0, column = 3)
upper_entry.grid(row=1, column = 3)
lower_entry.grid(row=2, column = 3)

e3.grid(row=0, column=5)
e4.grid(row=1, column=5)
samp_entry2.grid(row=0, column = 7)
upper_entry2.grid(row=1, column = 7)
lower_entry2.grid(row=2, column = 7)

ind_prop.grid(row=0, column = 9)
ind_up.grid(row=1, column = 9)
ind_low.grid(row=2, column = 9)

tk.Button(master, text='Quit', command=master.quit).grid(row=3, column=0, pady=4)
tk.Button(master, text='Perform inference', command=init_inf).grid(row=3, column=1, pady=4)

master.mainloop()