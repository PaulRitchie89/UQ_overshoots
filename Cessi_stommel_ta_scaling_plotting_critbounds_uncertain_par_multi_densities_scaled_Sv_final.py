# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:01:43 2024

@author: Paul Ritchie

Script to plot Figure 7.  Constraining uncertainty in diffusive timescale
minimises uncertainty in overshoot boundary separating tipping and avoiding
tipping.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from matplotlib import rc
import seaborn as sns
from scipy import stats

fontsize = 8
rc('font', **{'size' : fontsize})
rc('text', usetex=True)


## Model and distribution parameters
ta = 70 # Advective timescale (yrs)
alphaT = 1E-4 # Thermal expansion coefficient (/K)
alphaS = 7.6E-4 # Haline contraction coefficient (/psu)
S0 = 35 # Reference salinity (psu)
theta = 25 # Meridional temperature difference (K)
H = 4500 # Mean ocean depth (m)
V = 3.5E16 # Reference ocean volume (m**3)
Area = V/H # Ocean area (m**2)
beta = 365*24*3600 # Seconds in a year (s/yr)
gamma = 1E-6 # Conversion to Sv (Sv s / m**3)
fscaling = alphaT*theta*H*Area*gamma/(alphaS*S0*ta)/beta # Freshwater scaling (Sv)
x0 = 0 # Initial non-dimensional salinity
tdmin = 700 # Diffusive timescale corresponding to minimum threshold (yrs)
tdmax = 210 # Diffusive timescale corresponding to maximum threshold (yrs)
etasquaredmin = tdmin/ta


## Calculate lowest threshold in freshwater forcing
Amin = np.sqrt(1/9 - 1/(3*etasquaredmin))
muminusmin = 2/27 + 2/(3*etasquaredmin) + Amin*(1/3-1/etasquaredmin-Amin**2)


## Array of probability contour levels (note: reversed for diffusive timescale)
quantiles = np.array([0.99, 0.9, 0.75, 0.5])

## Probability contour levels for prior
tds = tdmax + (tdmin-tdmax)*quantiles
etasquareds = np.array(tds/ta)
## Probability contour levels for posterior (reading in data from Bayesian inference)
etasquareds_data = loadtxt("UQpostSample_y0_0K2_T5_etaP2_7K5_mu_0_noise0K01overSqrtEtaP2_numData501_priorUni_3_10_AIES_steps400_Nchains100.txt", comments="#", delimiter=",", unpack=False)
etasquareds2 = np.quantile(etasquareds_data,quantiles)


# Concatenate and order etasquared from the prior and posterior distributions
etasquared_concat1 = np.concatenate((etasquareds[etasquareds>etasquareds2[0]],etasquareds2))
etasquared_concat = np.concatenate((etasquared_concat1,etasquareds[etasquareds<etasquareds2[-1]]))

N = len(etasquared_concat)

## Array of peak freshwater forcing
Fmax2 = np.linspace(1*Area*gamma/beta/fscaling,2*Area*gamma/beta/fscaling,501)

# Initialise arrays
tesquared = np.zeros((len(etasquareds),len(Fmax2)))
tesquared2 = np.zeros((len(etasquareds),len(Fmax2)))

## Loop over all ratios of diffusive timescale and all peak
## freshwater forcing and calculate theoretical time over threshold.
for k in range(len(etasquareds)):
    for j in range(len(Fmax2)):
        ## Time over for prior
        A = np.sqrt(1/9 - 1/(3*etasquareds[k]))
        kappa = 3*A 
        Fminus = 2/27 + 2/(3*etasquareds[k]) + A*(1/3-1/etasquareds[k]-A**2)
        # If no overshoot set time over very large else calculate via inverse square law
        if Fmax2[j] <= Fminus:
            tesquared[k,j] = 100000
        else:    
            tesquared[k,j] = 4*(Fmax2[j]-muminusmin)/(kappa*((Fmax2[j]-Fminus)**2))

        ## Time over for posterior
        A = np.sqrt(1/9 - 1/(3*etasquareds2[k]))
        kappa = 3*A 
        Fminus = 2/27 + 2/(3*etasquareds2[k]) + A*(1/3-1/etasquareds2[k]-A**2)
        # If no overshoot set time over very large else calculate via inverse square law
        if Fmax2[j] <= Fminus:
            tesquared2[k,j] = 100000
        else:    
            tesquared2[k,j] = 4*(Fmax2[j]-muminusmin)/(kappa*((Fmax2[j]-Fminus)**2))



## Initialising figure
fig, ax = plt.subplots(1,2,figsize=(7.007874,2.45))

ax[0].set_xlim(100,800)
ax[0].set_ylim(0,0.04)
ax[0].set_xlabel('Diffusive timescale (years)')
ax[0].set_ylabel('Probability density')
ax[0].set_yticks([0,0.01,0.02,0.03,0.04])

ax[1].set_xlim(1.1*Area*gamma/beta,1.5*Area*gamma/beta)
ax[1].set_ylim(0,5000)
ax[1].set_xlabel('Peak freshwater flux (Sv)')
ax[1].set_ylabel('Time over lowest threshold (years)')

sns.despine()

fig.tight_layout()

ax[0].text(0.87, 0.92, '\\textbf{(a)}', transform=ax[0].transAxes,size=12,fontsize=fontsize)
ax[1].text(0.87, 0.92, '\\textbf{(b)}', transform=ax[1].transAxes,size=12,fontsize=fontsize)

## Plotting panel (a)

# Plot prior for diffusive timescale (uniform in [210, 700])
ax[0].plot([tdmax,tdmax],[0,1/(tdmin-tdmax)],c=[136/255,46/255,114/255],lw=1,label='Prior')
ax[0].plot([tdmax,tdmin],[1/(tdmin-tdmax),1/(tdmin-tdmax)],c=[136/255,46/255,114/255],lw=1)
ax[0].plot([tdmin,tdmin],[0,1/(tdmin-tdmax)],c=[136/255,46/255,114/255],lw=1)

# Plot posterior for diffusive timescale (derived from Bayesian inference)
xs = np.linspace(100, 800, 500)
kde = stats.gaussian_kde(etasquareds_data*ta)
ax[0].plot(xs, kde.pdf(xs), c=[78/255,178/255,101/255],lw=1,label='Posterior')

ax[0].legend(frameon=False,loc='upper left')


## Plotting panel (b)
counter = 0
counter3 = 0

alphas = [0.2,0.4,0.6,0.4,0.2]

while etasquared_concat[counter+1] != etasquareds2[0]:
    ax[1].fill_between(fscaling*Fmax2,ta*np.sqrt(tesquared[counter,:]),ta*np.sqrt(tesquared[counter+1,:]),color=[136/255,46/255,114/255],alpha=alphas[counter],edgecolor='none')
    counter = counter+1

ax[1].fill_between(fscaling*Fmax2,ta*np.sqrt(tesquared[counter,:]),ta*np.sqrt(tesquared2[0,:]),color=[136/255,46/255,114/255],alpha=alphas[counter],edgecolor='none')

for counter2 in range(len(etasquareds2)-1):
    ax[1].fill_between(fscaling*Fmax2,ta*np.sqrt(tesquared2[counter2,:]),ta*np.sqrt(tesquared2[counter2+1,:]),color=[78/255,178/255,101/255],alpha=alphas[counter2],edgecolor='none')

ax[1].fill_between(fscaling*Fmax2,ta*np.sqrt(tesquared2[-1,:]),ta*np.sqrt(tesquared[-counter3-1,:]),color=[136/255,46/255,114/255],alpha=alphas[2],edgecolor='none')
