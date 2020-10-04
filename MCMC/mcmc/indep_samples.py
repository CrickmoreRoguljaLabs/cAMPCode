#### compute credible intervals for two independent simultaneous trials with a Jeffreys prior SCT 06/10/2016

import pymc
import numpy as np
import matplotlib.pyplot as plt

def indep_samples(x1,n1,x2,n2):
	class experimental_data(object):
		
		def __init__(self,x1,n1,x2,n2):
			self.data1 = np.hstack( (np.ones((x1,)) , np.zeros((n1-x1,))) )
			self.data2 = np.hstack( (np.ones((x2,)) , np.zeros((n2-x2,))) )

	### for testing purposes
	#example = True

	data = experimental_data(x1,n1,x2,n2)

	#if example:
	#	example_data1 = np.hstack( (np.ones((15,)) , np.zeros((20,))) )
	#	example_data2 = np.hstack( (np.ones((16,)) , np.zeros((23,))) )
	#	sim_data_size = 14

	#	data1 = example_data1
	#	data2 = example_data2

	p1_val = np.mean(data.data1)
	p2_val = np.mean(data.data2)
	ind_val = p1_val+p2_val-p1_val*p2_val
	print "P1 = " + str(p1_val)

	print "P2 = " + str(p2_val)

	print "Independence = " + str(ind_val)

	p1 = pymc.Beta('p1',alpha=0.5,beta=0.5)
	p2 = pymc.Beta('p2',alpha=0.5,beta=0.5)

	x1 = pymc.Binomial('x',n=len(data.data1),p=p1,value=np.sum(data.data1),observed=True)
	x2 = pymc.Binomial('x',n=len(data.data2),p=p2,value=np.sum(data.data2),observed=True)

	@pymc.deterministic
	def ind_assump(p1=p1,p2=p2):
		return p1+p2-p1*p2

	return locals()
	#@pymc.deterministic
	#def sim():
	#	sim_data = pymc.Binomial('sim',n=sim_data_size, p=ind_assump)
	#	return sim_data