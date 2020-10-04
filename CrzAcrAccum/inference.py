### Pymc2 implementation of a hierarchical model generating copulation duration, with two possible distributions from which the data is sampled:
# Crz prevented from being active, and Crz permitted to be active. Each is modeled as a Gaussian.
# SCT 05/17/2017
import numpy as np
import pandas as pd
import scipy.stats
#import matplotlib.pyplot as plt
import pymc
import curate_data
#import tkinter as tk
#from tkinter import filedialog

def sample_stats(sample):
	# return the mean, variance, and size of a sample
	mean = np.mean(sample)
	var = np.var(sample)
	size = np.shape(sample)[0]
	sem = np.sqrt(var)/np.sqrt(size)
	return mean, var, sem, size

## Location of spreadsheet. Column names: Genotype, Condition, Duration
data_location = "/Users/stephen/Desktop/cAMP Paper/Window Accumulation Experiments Charlotte.xlsx"
genotype = "Crz>ACR"

off_at_10, df_dict = curate_data.curate_data(data_location, genotype=genotype)

# Values needed to establish the prior on the two normal distributions for copulation duration
mean_10, var_10, sem_10, size_10 = sample_stats(off_at_10)
variances = var_10

# Establish the priors for the off at 10 condition, estimated from the experiments themselves
sigmas = pymc.Chi2('sigma', np.array([size_10-1]),size=1)
mus = pymc.Normal('Mu', mu=np.array([mean_10]), tau=[1.0/sem_10],size=1)

@pymc.deterministic
def tau(sigmas=sigmas):
    return 1./(variances*sigmas)

print(df_dict["20s window"])
term_time = df_dict["20s window"]["mating duration (min) truncated at 60 min "]

switch_lag = pymc.Normal('Sampled data', mu = mu, tau=tau)

@pymc.deterministic
def obs(switches=switch_lag):
	return term_time - switches

model = pymc.Model([obs, switch_lag, mus, sigmas])
mcmc = pymc.MCMC(model)
num_expts = 70000
burn_in = 3000
skip = 5
mcmc.sample(num_expts,burn_in,skip) 
# check the acceptance rates
ps = mcmc.trace("obs")[:]
p_sorted = np.zeros(ps.shape)
for k in range(ps.shape[1]):
	p_sorted[:,k] = np.sort(ps[:,k])

#pd.traceplot(ps)
######
# REPORTING INFERENCE ON THE COMMAND LINE

writer = pd.ExcelWriter('InferenceOut.xlsx', engine='xlsxwriter')
output_df = pd.DataFrame()
# Report the credible intervals, 68%
lower_bound = p_sorted[int(.16*(num_expts-burn_in)/skip),:]
output_df["lower bound"] = lower_bound
median = p_sorted[int(.5*(num_expts-burn_in)/skip),:]
output_df["median"] = median
upper_bound = p_sorted[int(.84*(num_expts-burn_in)/skip),:]
output_df["upper_bound"] = upper_bound

output_df.to_excel(writer, sheet_name='Summary of statistics')


# print("Credible interval for p " + str([lower_bound,median,upper_bound]))
# m = mcmc.trace("Mu")[:]
# mu = np.sort(m[:][:,0])
# lower_bound = mu[int(.16*(num_expts-burn_in)/skip)]
# mu1_median = mu[int(.5*(num_expts-burn_in)/skip)]
# upper_bound = mu[int(.84*(num_expts-burn_in)/skip)]
# print("Credible interval for mu1 " + str([lower_bound,mu1_median,upper_bound]))
# mu = np.sort(m[:][:,1])
# lower_bound = mu[int(.16*(num_expts-burn_in)/skip)]
# mu2_median = mu[int(.5*(num_expts-burn_in)/skip)]
# upper_bound = mu[int(.84*(num_expts-burn_in)/skip)]
# print("Credible interval for mu2 " + str([lower_bound,mu2_median,upper_bound]))
# s = mcmc.trace("sigma")[:]
# sig = np.sqrt(np.sort((1.0/s[:])[:,0])*variances[0])
# lower_bound = sig[int(.16*(num_expts-burn_in)/skip)]
# sig1_median = sig[int(.5*(num_expts-burn_in)/skip)]
# upper_bound = sig[int(.84*(num_expts-burn_in)/skip)]
# print("Credible interval for sigma 1" + str([lower_bound,sig1_median,upper_bound]))
# sig = np.sqrt(np.sort((1.0/s[:])[:,1])*variances[1])
# lower_bound = sig[int(.16*(num_expts-burn_in)/skip)]
# sig2_median = sig[int(.5*(num_expts-burn_in)/skip)]
# upper_bound = sig[int(.84*(num_expts-burn_in)/skip)]
# print("Credible interval for sigma 2" + str([lower_bound,sig2_median,upper_bound]))


writer.save()

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

### The separatrix between "long" and "normal" matings

#print(solve(mu1_median,mu2_median,np.sqrt(sig1_median),np.sqrt(sig2_median)))

#plt.scatter(np.zeros_like(mu),mu)
#plt.show()