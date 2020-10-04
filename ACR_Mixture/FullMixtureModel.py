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
data_location = "/Users/stephen/Desktop/cAMP Paper/Crz_ACR_Timing_for_cAMP.xlsx"
#data_location = "~/Desktop/FruitFly/ACR_Mixture/Crz_ACR_Timing.xlsx"
#data_location = "~/Desktop/FruitFly/ACR_Mixture/Crz_ACR Windows.xlsx"
#genotype = "Crz>CaMKII-RNAi, ACR"
#genotype = "Crz>D2,CaMKII-RNAi, ACR"
#genotype = "Crz>D2,CaMKII-WT, ACR"
#genotype ="Crz>WT-CaMKII,ACR"
#sheet_name="CaMKII OE"
sheet_name="Sheet1"
genotype="Crz>ACR"
#sheet_name="OE"
no_light, at_start, data = curate_data.curate_data(data_location, genotype=genotype, sheet=sheet_name)

# Values needed to establish the prior on the two normal distributions for copulation duration
mean_no_light, var_no_light, sem_no_light, size_no_light = sample_stats(no_light)
mean_start,var_start,sem_start,size_start = sample_stats(at_start)
variances = np.array([var_start, var_no_light])

# Integer indices for each condition
condition_cat = pd.Categorical(data['Condition'])
conditions = condition_cat.codes
num_conds = len(set(conditions))

# Establish the priors for the no light and with light conditions, estimated from the experiments themselves
sigmas = pymc.Chi2('sigma', np.array([size_start-1,size_no_light-1]),size=2)
mus = pymc.Normal('Mu', mu=np.array([mean_start,mean_no_light]), tau=[1.0/sem_start,1.0/sem_no_light],size=2)

# Jeffrey's prior for the probability of Crz having gone off at that specific time
p  = pymc.Beta('p', alpha=0.5, beta=0.5, size=num_conds)

@pymc.deterministic
def prob(conditions=conditions, p=p):
	return [(1-p[c],p[c]) for c in conditions]

#print len(data["Duration"])
# samples fall in each category with probability prob for being in the "lights off" state
category = pymc.Categorical("category", p=prob,size=len(data["Duration"]))

## Deterministic variables describing which mu / tau / p to use dependent on your condition or category

@pymc.deterministic
def mu(category=category, mus=mus):
    return mus[category]

@pymc.deterministic
def tau(category=category, sigmas=sigmas):
    return 1./(variances[category]*sigmas[category])

observations = pymc.Normal('Sampled data', mu = mu, tau=tau,value=data["Duration"],observed=True)
model = pymc.Model([observations, category, p, mus, sigmas])
mcmc = pymc.MCMC(model)
num_expts = 70000
burn_in = 3000
skip = 5
mcmc.sample(num_expts,burn_in,skip) 
# check the acceptance rates
ps = mcmc.trace("p")[:]
p_sorted = np.zeros(ps.shape)
for k in range(ps.shape[1]):
	p_sorted[:,k] = np.sort(ps[:,k])

#pd.traceplot(ps)
######
# REPORTING INFERENCE ON THE COMMAND LINE

print condition_cat.categories

writer = pd.ExcelWriter('MixtureModelOutput_%s.xlsx'%(genotype,), engine='xlsxwriter')
output_df = pd.DataFrame(index=condition_cat.categories)
# Report the credible intervals, 68%
lower_bound = p_sorted[int(.16*(num_expts-burn_in)/skip),:]
output_df["lower bound"] = lower_bound
median = p_sorted[int(.5*(num_expts-burn_in)/skip),:]
output_df["median"] = median
upper_bound = p_sorted[int(.84*(num_expts-burn_in)/skip),:]
output_df["upper_bound"] = upper_bound

output_df.to_excel(writer, sheet_name='Summary of statistics')


print("Credible interval for p " + str([lower_bound,median,upper_bound]))
m = mcmc.trace("Mu")[:]
mu = np.sort(m[:][:,0])
lower_bound = mu[int(.16*(num_expts-burn_in)/skip)]
mu1_median = mu[int(.5*(num_expts-burn_in)/skip)]
upper_bound = mu[int(.84*(num_expts-burn_in)/skip)]
print("Credible interval for mu1 " + str([lower_bound,mu1_median,upper_bound]))
mu = np.sort(m[:][:,1])
lower_bound = mu[int(.16*(num_expts-burn_in)/skip)]
mu2_median = mu[int(.5*(num_expts-burn_in)/skip)]
upper_bound = mu[int(.84*(num_expts-burn_in)/skip)]
print("Credible interval for mu2 " + str([lower_bound,mu2_median,upper_bound]))
s = mcmc.trace("sigma")[:]
sig = np.sqrt(np.sort((1.0/s[:])[:,0])*variances[0])
lower_bound = sig[int(.16*(num_expts-burn_in)/skip)]
sig1_median = sig[int(.5*(num_expts-burn_in)/skip)]
upper_bound = sig[int(.84*(num_expts-burn_in)/skip)]
print("Credible interval for sigma 1" + str([lower_bound,sig1_median,upper_bound]))
sig = np.sqrt(np.sort((1.0/s[:])[:,1])*variances[1])
lower_bound = sig[int(.16*(num_expts-burn_in)/skip)]
sig2_median = sig[int(.5*(num_expts-burn_in)/skip)]
upper_bound = sig[int(.84*(num_expts-burn_in)/skip)]
print("Credible interval for sigma 2" + str([lower_bound,sig2_median,upper_bound]))


writer.save()

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

### The separatrix between "long" and "normal" matings

print(solve(mu1_median,mu2_median,np.sqrt(sig1_median),np.sqrt(sig2_median)))

#plt.scatter(np.zeros_like(mu),mu)
#plt.show()