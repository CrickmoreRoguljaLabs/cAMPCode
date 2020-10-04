#### Defines a model for two independent Bernoulli processes with the Jeffreys prior on the probability parameter p
#### Returns sampled probability parameter
#### SCT 06/10/2016

import pymc
import numpy as np
import matplotlib.pyplot as plt

def indep_samples(x1,n1,x2,n2):

	# clumsy holdover from a previous implementation that I've been too lazy to rewrite.
	class experimental_data(object):
		
		def __init__(self,x1,n1,x2,n2):
			self.data1 = np.hstack( (np.ones((x1,)) , np.zeros((n1-x1,))) )
			self.data2 = np.hstack( (np.ones((x2,)) , np.zeros((n2-x2,))) )

	# why do I do n1 here and number of failures in mcmc_indep?? That was pretty dumb of me
	data = experimental_data(x1,n1,x2,n2)

	p1_val = np.mean(data.data1)
	p2_val = np.mean(data.data2)
	ind_val = p1_val+p2_val-p1_val*p2_val
	print "P1 = " + str(p1_val)

	print "P2 = " + str(p2_val)

	print "Independence = " + str(ind_val)

	prior1 = pymc.Beta('p1',alpha=0.5,beta=0.5)
	prior2 = pymc.Beta('p2',alpha=0.5,beta=0.5)

	x1 = pymc.Binomial('x',n=len(data.data1),p=prior1,value=np.sum(data.data1),observed=True)
	x2 = pymc.Binomial('x',n=len(data.data2),p=prior2,value=np.sum(data.data2),observed=True)

	@pymc.deterministic
	def ind_assump(p1=p1,p2=p2):
		return p1+p2-p1*p2

	return locals()