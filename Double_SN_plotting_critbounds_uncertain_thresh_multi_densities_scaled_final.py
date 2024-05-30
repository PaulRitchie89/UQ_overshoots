# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:10:02 2024

@author: Paul Ritchie

Script to plot Figure 2. Constraining uncertainty in threshold location minimises
uncertainty in overshoot boundary separating tipping and avoiding tipping
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from scipy import stats


fontsize = 8
rc('font', **{'size' : fontsize})
rc('text', usetex=True)

## Initialising figure
fig, ax = plt.subplots(1,2,figsize=(7.007874,2.45))

ax[0].set_xlim(2-0.05,2+0.35)
ax[0].set_ylim(-0.1,21)
ax[1].set_xlim(2,2.4)
ax[1].set_ylim(0,40)

ax[0].set_xlabel('Threshold location')
ax[0].set_ylabel('Probability density')
ax[1].set_xlabel('Peak external forcing')
ax[1].set_ylabel('Time over lowest possible threshold')

sns.despine()

fig.tight_layout()

ax[0].text(0.87, 0.92, '\\textbf{(a)}', transform=ax[0].transAxes,size=12,fontsize=fontsize)
ax[1].text(0.87, 0.92, '\\textbf{(b)}', transform=ax[1].transAxes,size=12,fontsize=fontsize)

## System and distribution parameters
a0 = 1
kappa = 1
deltamin = 2
width = 0.3
deltamean = 2.1
deltasigma = 0.02

## Plotting panel (a)

# Plot initial distribution for threshold location (uniform in [2, 2.3])
ax[0].plot([1.9,deltamin],[0,0],c=[136/255,46/255,114/255],lw=1,label='Initial')
ax[0].plot([deltamin,deltamin],[0,1/width],c=[136/255,46/255,114/255],lw=1)
ax[0].plot([deltamin,deltamin+width],[1/width,1/width],c=[136/255,46/255,114/255],lw=1)
ax[0].plot([deltamin+width,deltamin+width],[0,1/width],c=[136/255,46/255,114/255],lw=1)
ax[0].plot([deltamin+width,2.4],[0,0],c=[136/255,46/255,114/255],lw=1)

# Plot knowledge-based distribution for threshold location (Normal, mean=2.1, sd=0.02)
xs = np.linspace(1.9,2.4,5000)
dist = stats.norm(loc=deltamean, scale=deltasigma)

ax[0].plot(xs, dist.pdf(xs), c=[78/255,178/255,101/255],lw=1,label='Knowledge-based')
ax[0].legend(frameon=False,loc='center right')


## Array of peak external forcing
mushifts = deltamin + np.linspace(0.001, 0.4, 400)

## List of probability contour levels
quantiles = [0.01, 0.1, 0.25, 0.5]

## Probability contour levels for initial distribution of threshold location (uniform in [2, 2.3])
deltas = deltamin + width*np.array(quantiles)

## Probability contour levels for knowledge-based distribution of threshold location (Normal, mean=2.1, sd=0.02)
deltas2 = dist.ppf(quantiles)

# Concatenate and order delta from the 2 distributions
delta_concat1 = np.concatenate((deltas[deltas<deltas2[0]],deltas2))
delta_concat = np.concatenate((delta_concat1,deltas[deltas>deltas2[-1]]))

N = len(delta_concat)

# Initialise arrays 
tesquared = np.zeros((len(deltas),len(mushifts)))
tesquared2 = np.zeros((len(deltas2),len(mushifts)))

## Loop over all threshold locations and all peak external forcing and
## calculate theoretical time over lowest threshold. If peak doesn't exceed
##  threshold, set time over large 
for k in range(len(deltas)):
    for j in range(len(mushifts)):
        if mushifts[j] <= deltas[k]:
            tesquared[k,j] = 5000
        else:
            tesquared[k,j] = 4*(mushifts[j]-deltamin)/((a0**2)*kappa*((mushifts[j]-deltas[k])**2))
        if mushifts[j] <= deltas2[k]:
            tesquared2[k,j] = 5000
        else:
            tesquared2[k,j] = 4*(mushifts[j]-deltamin)/((a0**2)*kappa*((mushifts[j]-deltas2[k])**2))


## Plotting panel (b)
counter = 0
counter3 = 0

alphas = [0.2,0.4,0.6]

while delta_concat[counter+1] != deltas2[0]:
    ax[1].fill_between(mushifts,np.sqrt(tesquared[counter,:]),np.sqrt(tesquared[counter+1,:]),color=[136/255,46/255,114/255],alpha=alphas[counter],edgecolor='none')
    counter = counter+1

ax[1].fill_between(mushifts,np.sqrt(tesquared[counter,:]),np.sqrt(tesquared2[0,:]),color=[136/255,46/255,114/255],alpha=alphas[counter],edgecolor='none')

for counter2 in range(len(deltas2)-1):
    ax[1].fill_between(mushifts,np.sqrt(tesquared2[counter2,:]),np.sqrt(tesquared2[counter2+1,:]),color=[78/255,178/255,101/255],alpha=alphas[counter2],edgecolor='none')

while counter+counter2+counter3<N-4:
    ax[1].fill_between(mushifts,np.sqrt(tesquared[-counter3-2,:]),np.sqrt(tesquared[-counter3-1,:]),color=[136/255,46/255,114/255],alpha=alphas[-counter3-1],edgecolor='none')
    counter3 = counter3+1

ax[1].fill_between(mushifts,np.sqrt(tesquared2[-1,:]),np.sqrt(tesquared[-counter3-1,:]),color=[136/255,46/255,114/255],alpha=alphas[-counter3-1],edgecolor='none')





