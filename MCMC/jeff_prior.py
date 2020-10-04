#### compute credible intervals with a Jeffreys prior SCT 06/10/2016

import pymc
import numpy as np
import matplotlib.pyplot as plt

### for testing purposes
example = True

example_data = np.hstack( (np.ones((5,)) , np.zeros((8,))) )

print example_data

if example:
	data = example_data

prior = pymc.Beta('prior',alpha=0.5,beta=0.5)

x = pymc.Binomial('x',n=len(data),p=prior,value=np.sum(data),observed=True)

#likelihood = pymc.Binomial()