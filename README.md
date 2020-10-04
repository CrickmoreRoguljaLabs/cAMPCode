# cAMPCode
Code for Thornquist, Pitsch, et al.

Organized into a few subsections:

## ACR_Mixture
Code for determining long vs. normal duration matings using a Bayesian Gaussian Mixture Model on copulation durations. Makes use of the pymc package (can be installed through pip or conda).

## CrzACRAccum
Model for inferring eruption time on copulation duration data used in Figure 7. Uses the PyStan package. Includes output .pdf plots

## Flymage
Code for interpreting and analyzing FLIMage! files (.flim) and converting them to dF/F traces or fluorescence lifetime traces.

## LightPad
Code and PCB design for controlling Artograph light pads with an Arduino, a power supply, and a PCB. Uses MOSFET transistors to gate power supply to each pad.

## MCMC
Same as code from Thornquist et al., Neuron 2020 about CaMKII; used to estimate posterior distributions (and credible intervals) when plotting proportions. Uses the pymc package.
