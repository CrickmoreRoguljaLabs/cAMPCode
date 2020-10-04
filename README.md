# cAMPCode
Code for Thornquist, Pitsch, et al.

All code written by SCT (sometimes it's embarrassingly bad, but here it is). Most Python 2.7 code should be compatible with Python 3.x environments, though I haven't systematically checked.

Organized into a few subsections:

## ACR_Mixture
Code for determining long vs. normal duration matings using a Bayesian Gaussian Mixture Model on copulation durations. Makes use of the pymc package (can be installed through pip or conda). Written in a Python 2.7 environment

## CrzACRAccum
Model for inferring eruption time on copulation duration data used in Figure 7. Uses the PyStan package. Includes output .pdf plots. Written in a Python 2.7 environment.

## Flymage
Code for interpreting and analyzing FLIMage! files (.flim) and converting them to dF/F traces or fluorescence lifetime traces. Written in a Python 3.6 environment

## LightPad
Code and PCB design for controlling Artograph light pads with an Arduino, a power supply, and a PCB. Uses MOSFET transistors to gate power supply to each pad.

## MCMC
Same as code from Thornquist et al., Neuron 2020 about CaMKII; used to estimate posterior distributions (and credible intervals) when plotting proportions. Uses the pymc package. Written in a Python 2.7 environment.
