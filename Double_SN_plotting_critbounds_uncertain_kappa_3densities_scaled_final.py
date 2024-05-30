# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:10:37 2024

@author: Paul Ritchie

Script to plot Figure 4. Constraining uncertainty in linear restoring force does
not necessarily reduce uncertainty in overshoot boundary separating tipping and
avoiding tipping.
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

ax[0].set_xlim(-0.5,7.5)
ax[0].set_ylim(-0.005,1.7)
ax[1].set_xlim(0,0.15)
ax[1].set_ylim(2,10)

ax[0].set_xlabel('Restoring force proportionality factor')
ax[0].set_ylabel('Probability density')
ax[1].set_xlabel('Distance of overshoot')
ax[1].set_ylabel('Time over threshold')

sns.despine()

fig.tight_layout()

ax[0].text(0.87, 0.92, '\\textbf{(a)}', transform=ax[0].transAxes,size=12,fontsize=fontsize)
ax[1].text(0.87, 0.92, '\\textbf{(b)}', transform=ax[1].transAxes,size=12,fontsize=fontsize)

## System and distribution parameters
a0 = 1
delta = 2
kappamin = 0.25
kappamin2 = 3.5
width = 3
kappamean = 1
kappasigma = 0.25


## Plotting panel (a)

# Plot initial distribution for linear restoring force proportionality constant (uniform in [0.25, 3.25])
ax[0].plot([-0.6,kappamin],[0,0],c=[136/255,46/255,114/255],lw=1,label='Initial')
ax[0].plot([kappamin,kappamin],[0,1/width],c=[136/255,46/255,114/255],lw=1)
ax[0].plot([kappamin,kappamin+width],[1/width,1/width],c=[136/255,46/255,114/255],lw=1)
ax[0].plot([kappamin+width,kappamin+width],[0,1/width],c=[136/255,46/255,114/255],lw=1)
ax[0].plot([kappamin+width,10],[0,0],c=[136/255,46/255,114/255],lw=1)

# Plot alternative distribution for linear restoring force proportionality constant (uniform in [3.5, 65])
ax[0].plot([-0.6,kappamin2],[0,0],c='b',lw=1,label='Alternative')
ax[0].plot([kappamin2,kappamin2],[0,1/width],c='b',lw=1)
ax[0].plot([kappamin2,kappamin2+width],[1/width,1/width],c='b',lw=1)
ax[0].plot([kappamin2+width,kappamin2+width],[0,1/width],c='b',lw=1)
ax[0].plot([kappamin2+width,10],[0,0],c='b',lw=1)

# Plot knowledge-based distribution for linear restoring force proportionality constant (Normal, mean=1, sd=0.25)
xs = np.linspace(-0.5,7.5,5000)
dist = stats.norm(loc=kappamean, scale=kappasigma)

ax[0].plot(xs, dist.pdf(xs), c=[78/255,178/255,101/255],lw=1,label='Knowledge-based')
ax[0].legend(frameon=False,loc='center right')


## Array of peak external forcing
mushifts = delta + np.linspace(0.0001, 0.15, 3000)

## List of probability contour levels (note: reversed for kappa)
quantiles = [0.99, 0.9, 0.75, 0.5]

## Probability contour levels for initial distribution
kappas = kappamin+width*np.array(quantiles)
## Probability contour levels for alternative distribution
kappas2 = kappamin2+width*np.array(quantiles)
## Probability contour levels for knowledge-based distribution
kappas3 = dist.ppf(quantiles)

# Concatenate and order kappa from the initial and knowledge-based distributions
kappa_concat1 = np.concatenate((kappas[kappas>kappas3[0]],kappas3))
kappa_concat = np.concatenate((kappa_concat1,kappas[kappas<kappas3[-1]]))

N = len(kappa_concat)

# Initialise arrays
tesquared = np.zeros((len(kappas),len(mushifts)))
tesquared2 = np.zeros((len(kappas2),len(mushifts)))
tesquared3 = np.zeros((len(kappas3),len(mushifts)))

## Loop over all linear restoring rate proportionality constants and all peak
## external forcing and calculate theoretical time over threshold.
for k in range(len(kappas)):
    for j in range(len(mushifts)):
        tesquared[k,j] = 4/((a0**2)*kappas[k]*((mushifts[j]-delta)))
        tesquared2[k,j] = 4/((a0**2)*kappas2[k]*((mushifts[j]-delta)))
        tesquared3[k,j] = 4/((a0**2)*kappas3[k]*((mushifts[j]-delta)))




## Plotting panel (b)
alphas = [0.2,0.4,0.6,0.8]

for counter in range(len(kappas)-1):
    ax[1].fill_between(mushifts-delta,np.sqrt(tesquared2[counter,:]),np.sqrt(tesquared2[counter+1,:]),color='b',alpha=alphas[counter],edgecolor='none')

counter = 0
counter3 = 0

for counter2 in range(len(kappas3)-1):
    ax[1].fill_between(mushifts-delta,np.sqrt(tesquared3[counter2,:]),np.sqrt(tesquared3[counter2+1,:]),color=[78/255,178/255,101/255],alpha=alphas[counter2],edgecolor='none')

while kappa_concat[counter+1] != kappas3[0]:
    ax[1].fill_between(mushifts-delta,np.sqrt(tesquared[counter,:]),np.sqrt(tesquared[counter+1,:]),color=[136/255,46/255,114/255],alpha=alphas[counter],edgecolor='none')
    counter = counter+1

if kappas3[0]>kappas2[-1]:
    ax[1].fill_between(mushifts-delta,np.sqrt(tesquared[counter,:]),np.sqrt(tesquared3[0,:]),color=[136/255,46/255,114/255],alpha=alphas[counter],edgecolor='none')
    

    while counter+counter2+counter3<N-4:
        ax[1].fill_between(mushifts-delta,np.sqrt(tesquared[-counter3-2,:]),np.sqrt(tesquared[-counter3-1,:]),color=[136/255,46/255,114/255],alpha=alphas[-counter3-1],edgecolor='none')
        counter3 = counter3+1
    if kappas3[-1]>kappas2[-1]:
        ax[1].fill_between(mushifts-delta,np.sqrt(tesquared3[-1,:]),np.sqrt(tesquared[-counter3-1,:]),color=[136/255,46/255,114/255],alpha=alphas[-counter3-1],edgecolor='none')
