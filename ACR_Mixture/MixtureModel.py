# Testing and troubleshooting
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import pymc

def sample_stats(sample):
	# return the mean, variance, and size of a sample
	mean = np.mean(sample)
	var = np.var(sample)
	size = np.shape(sample)[0]
	sem = np.sqrt(var)/np.sqrt(size)
	return mean, var, sem, size
#"""
# TA,ACR
no_light = np.array([17,
21,
19,
18,
21,
20,
22,
24,
16,
21,
18,
22,
22,
26])
#"""

"""
# ACR alone

no_light = np.array([18,
23,
27,
15,
22,
25,
27,
25,
28,
23])
"""
# return basic stats on the no_light condition
(mean_no_light, var_no_light, sem_no_light, size_no_light) = sample_stats(no_light)

# likewise for light at start
#"""
#TA, ACR
light_at_start = np.array([152,
161,
203,
216,
135,
196,
234,
128,
134,
143,
166,
193])
"""
"""
new_data = np.array([19,
20,
22,
182,
110,
22,
249,
16,
18,
19,
17,
121,
187,
284,
17,
24,
191,
237,
166,
19,
29,
21,
22,
24,
26,
23,
28])
#"""
"""
#ACR alone

light_at_start = np.array([95,
99,
75,
100,
148,
159,
62,
84,
87,
114,
136,
174,
161,
156,
135,
182,
223,
209,
194])
"""

(mean_start, var_start, sem_start, size_start) = sample_stats(light_at_start)
"""
#3 min
new_data = np.array([19,
20,
22,
182,
110,
22,
249,
16,
18,
19,
17,
121,
187,
284])
"""

"""
#5 min
new_data = np.array([12,
20,
19,
19,
23,
22,
19,
19,
158,
20,
19,
20,
23,
21])
"""
"""
# ACR alone 5 min
new_data = np.array([214,
183,
198,
174,
125,
97,
143,
131,
166,
75,
224,
109])
"""

"""
# TA, ACR 7 min
new_data = np.array([25,
20,
18,
17])
"""
"""
# ACR alone 7 min
new_data = np.array([17,
21,
23,
18,
22,
18,
23,
16,
118,
147,
26,
28])
"""
"""
# ACR alone 10 min
new_data = np.array([18,
22,
19,
16,
23,
18,
16])
"""
nsamples = new_data.shape[0]

variances = np.array([var_no_light,var_start])
print np.sqrt(variances)
# Establish the priors for the no light and with light conditions, estimated from the experiments themselves
sigmas = pymc.Chi2('sigma', np.array([size_no_light-1,size_start-1]),size=2)
mus = pymc.Normal('Mu', mu=np.array([mean_no_light,mean_start]), tau=[1.0/sem_no_light,1.0/sem_start],size=2)

# Jeffrey's prior for the probability of Crz having gone off at that specific time
p  = pymc.Beta('p', alpha=0.5*np.ones((1,num_conds)), beta=0.5*np.ones((1,num_conds)))

# select nsamples to fall in each category with probability p for being in the "lights off" state
category = pymc.Categorical("category", [p, 1 - p], size=nsamples)

## Deterministic variables describing which mu / tau / p to use dependent on your condition or category

@pymc.deterministic
def mu(category=category, mus=mus):
    return mus[category]

@pymc.deterministic
def tau(category=category, sigmas=sigmas):
    return 1./(variances[category]*sigmas[category])

observations = pymc.Normal('Sampled data', mu = mu, tau=tau,value=new_data,observed=True)
model = pymc.Model([observations, category, p, mus, sigmas])
mcmc = pymc.MCMC(model)
num_expts = 20000
burn_in = 8000
skip = 5
mcmc.sample(num_expts,burn_in,skip) 
# check the acceptance rates
ps = mcmc.trace("p")[:]
ps = np.sort(ps)

# Report the credible intervals, 68%
lower_bound = ps[int(.16*(num_expts-burn_in)/skip)]
median = ps[int(.5*(num_expts-burn_in)/skip)]
upper_bound = ps[int(.84*(num_expts-burn_in)/skip)]
print("Credible interval for p " + str([lower_bound,median,upper_bound]))
m = mcmc.trace("Mu")[:]
mu = np.sort(m[:][:,0])
print mu.shape
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
sig = np.sort(s[:][:,0])
lower_bound = sig[int(.16*(num_expts-burn_in)/skip)]
sig1_median = sig[int(.5*(num_expts-burn_in)/skip)]
upper_bound = sig[int(.84*(num_expts-burn_in)/skip)]
print("Credible interval for sigma 1" + str([lower_bound,sig1_median,upper_bound]))
sig = np.sort(s[:][:,1])
lower_bound = sig[int(.16*(num_expts-burn_in)/skip)]
sig2_median = sig[int(.5*(num_expts-burn_in)/skip)]
upper_bound = sig[int(.84*(num_expts-burn_in)/skip)]
print("Credible interval for sigma 2" + str([lower_bound,sig2_median,upper_bound]))

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

print(solve(mu1_median,mu2_median,np.sqrt(sig1_median),np.sqrt(sig2_median)))

#plt.scatter(np.zeros_like(mu),mu)
#plt.show()