import numpy as numpy
import matplotlib.pyplot as plt
import pymc
import jeff_prior

S = pymc.MCMC(jeff_prior, db='pickle')
S.sample(iter=100000, burn=5000, thin=2)
pymc.Matplot.plot(S)
