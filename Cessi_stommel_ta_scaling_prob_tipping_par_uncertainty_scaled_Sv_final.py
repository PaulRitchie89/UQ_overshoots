# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:46:05 2024

@author: Paul Ritchie

Script to plot Figure 8.  Probability of avoiding tipping for a single
overshoot trajectory based on diffusive timescale distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import norm
from matplotlib.gridspec import GridSpec

fontsize = 8
rc('font', **{'size' : fontsize})
rc('text', usetex=True)

def Cessi(y,F,etasquared):

    """
    Stommel-Cessi AMOC model
    
    Inputs:
        y - state variable non-dimensionalised salinity 
        F - freshwater forcing profile
        etasquared - ratio of advective to diffusive timescale
    
    Returns:
        dydt - RHS for Stommel-Cessi AMOC model
    """
        
    dydt = (F - y*(1+etasquared*((1-y)**2))/etasquared)

    return(dydt)
    

    
def F(t,Fstart,Fstab,r,mu0,mu1):
    """
    Forcing profile from Huntingford et al. 2017 for freshwater
    
    Inputs:
        t - time
        Fstart - initial level of forcing
        Fstab - stabilising level of forcing
        r - proportional to initial rate of forcing
        mu0 - transition timescale for short term
        mu1 - transition timescale for long term
        
    Returns:
        Overshoot forcing profile
    """
    mu = mu0 + mu1*t
    gamma = r - mu0*(Fstab-Fstart)
    
    F = Fstart + gamma*t - (1 - np.exp(-mu*t))*(gamma*t - (Fstab-Fstart))
    
    return (F)


## Time parameters
tend = 120
tspan = [0,tend]
h = 0.01
t = np.arange(tspan[0],tspan[1]+h,h)
nt = len(t)

## Model parameters
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


## Non-diemnsionalised forcing parameters
Fstart = 0
Fstab = 1*Area*gamma/beta/fscaling
r = 0.01
mu0 = 0.005
mu1 = 0.0009
gamma2 = r - mu0*(Fstab-Fstart)
# Calculate time of maximum
tp = Fstab/(2*gamma2)-mu0/(4*mu1) +(np.sqrt((2*mu1*Fstab+mu0*gamma2)**2 + (8*mu1*(gamma2**2)))/(4*mu1*gamma2))
# Calculate curvature at maximum
d2t = (-(2*mu1*tp+mu0)*(gamma2-(gamma2*tp-Fstab)*(2*mu1*tp+mu0)+gamma2) - 2*mu1*(gamma2*tp-Fstab))*np.exp(-(mu0+mu1*tp)*tp)
# Calculate peak freshwater forcing
Fmax = F(tp,Fstart,Fstab,r,mu0,mu1)

## Set upper & lower bounds (both theory and numerical) for critical ratio
## of timescales for bisection method
etasquareda = 3
etasquaredb = 10
etasquareda_num = 3
etasquaredb_num = 10

## Iterate bisection step 100 times
for l in range(100):
    
    ## Theory
    
    # Determine mid point
    etasquaredc = (etasquareda+etasquaredb)/2

    # Calculate tipping threshold
    A = np.sqrt(1/9 - 1/(3*etasquaredc))
    xplus = 2/3+A
    kappa = 3*A 
    Fminus = 2/27 + 2/(3*etasquaredc) + A*(1/3-1/etasquaredc-A**2)
    
    # Inverse square law
    Theory = Fmax - Fminus - np.sqrt(-d2t/(2*kappa))
    
    # If tipping predicted (Theory>0) set mid point as new upper bound otherwise
    # set as new lower bound
    if Theory>0:
        etasquaredb = etasquaredc
    else:
        etasquareda = etasquaredc
    
    ## Numerical
    
    # Determine mid point
    etasquaredc_num = (etasquareda_num+etasquaredb_num)/2
    
    # Calculate fold bifurcation level in state variable to set threshold for tipping
    A = np.sqrt(1/9 - 1/(3*etasquaredc))        
    xplus = 2/3 + A
    
    # Initialise array
    X = np.zeros(len(t)+1)
    
    # Perform Forward Euler
    for i in range(len(t)):
        X[i+1] = X[i] + h*Cessi(X[i],F(t[i],Fstart,Fstab,r,mu0,mu1),etasquaredc_num)
        
    # If system tips set mid point as new upper bound otherwise set as new lower bound
    if np.max(X) > xplus:
        etasquaredb_num = etasquaredc_num
    else:
        etasquareda_num = etasquaredc_num


    
## Store critical ratio of timescales for overshoot trajectory considered
etasquaredcrit = etasquaredc
etasquaredcrit_num = etasquaredc_num

## Initialise arrays for mean and standard deviation for normal distribution of
## ratio of diffusive to advective timescale
mus = np.linspace(5,10,501)
sigmas = np.linspace(0.00000000001,150/70,101)

## Initialise arrays to store probability of avoiding tipping
Safe_prob_num = np.zeros((len(mus),len(sigmas)))
Safe_prob = np.zeros((len(mus),len(sigmas)))

## Determine probability of not tipping by evaluating cumulative probability density
## at critical ratio for all mean and standard deviations of normal distribution
for j in range(len(mus)):
    for k in range(len(sigmas)):
        Safe_prob_num[j,k] = norm.cdf(etasquaredcrit_num, mus[j], sigmas[k])*100
        Safe_prob[j,k] = norm.cdf(etasquaredcrit, mus[j], sigmas[k])*100




## Initialise figure
fig=plt.figure(figsize=(3.4252,4))
gs=GridSpec(2,2,height_ratios=[0.3,0.7],width_ratios=[0.85,0.15])
ax=fig.add_subplot(gs[0,:])
ax2=fig.add_subplot(gs[1,0])

ax.set_xlabel('Time (years)')
ax.set_ylabel('Freshwater flux\n(m/yr)')
ax.set_xlim(0,7000)
ax.set_ylim(0.9*Area/beta/1E6,1.45*Area/beta/1E6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax2.set_xlabel('Mean of diffusive timescale (years)')
ax2.set_ylabel('Standard deviation of diffusive\ntimescale (years)')
ax2.set_yticks([0,25,50,75,100,125,150])
plt.tight_layout()

ax.text(0.92, 0.92, '\\textbf{(a)}', transform=ax.transAxes,size=12,fontsize=fontsize)
ax2.text(0.91, 0.94, '\\textbf{(b)}', transform=ax2.transAxes,size=12,fontsize=fontsize, weight=1000)

## Determine the critical threshold in freshwater forcing for mean, and +/- 1
## standard deviation for selected normal distribution in ratio of timescales 
tdmean = 475
tdstd = 125
etasquared = etasquaredcrit_num
A = np.sqrt(1/9 - 1/(3*etasquared))
Fminuscrit = 2/27 + 2/(3*etasquared) + A*(1/3-1/etasquared-A**2)
etasquared = tdmean/ta
A = np.sqrt(1/9 - 1/(3*etasquared))
Fminusmean = 2/27 + 2/(3*etasquared) + A*(1/3-1/etasquared-A**2)
etasquared = (tdmean-tdstd)/ta
A = np.sqrt(1/9 - 1/(3*etasquared))
Fminuslowstd = 2/27 + 2/(3*etasquared) + A*(1/3-1/etasquared-A**2)
etasquared = (tdmean+tdstd)/ta
A = np.sqrt(1/9 - 1/(3*etasquared))
Fminushighstd = 2/27 + 2/(3*etasquared) + A*(1/3-1/etasquared-A**2)

## Plotting panel (a)
ax.fill_between(t*ta,fscaling*Fminuslowstd,fscaling*Fminushighstd,color='tab:orange',alpha=0.3,edgecolor='none')
ax.plot([t[0]*ta,t[-1]*ta],[fscaling*Fminusmean,fscaling*Fminusmean],color='tab:orange',lw=0.75)
ax.plot([t[0]*ta,t[-1]*ta],[fscaling*Fminuscrit,fscaling*Fminuscrit],ls='-',color='k',lw=0.75)
ax.plot(t*ta,fscaling*F(t,Fstart,Fstab,r,mu0,mu1),'b',lw=1)


## Plotting panel (b)
levels = np.array([0,1, 10, 25, 75, 90, 99,100])
levels3 = np.array([1, 99])
levels4 = np.array([10, 90])
levels5 = np.array([25, 75])

[MUS,SIGMAS] = np.meshgrid(ta*mus,ta*sigmas)
im = ax2.contourf(MUS,SIGMAS,100-Safe_prob_num.T,levels=levels,vmin=0,vmax=100,cmap='coolwarm')

cbar_ax = fig.add_axes([0.8, 0.101, 0.03, 0.549])
cbar = fig.colorbar(im,cax=cbar_ax,spacing='proportional')
cbar.set_ticks(np.arange(0,101,10))
cbar.set_ticklabels(np.arange(0,101,10))
cbar.set_label('Probability of tipping')

ax2.contour(MUS,SIGMAS,100-Safe_prob.T,levels=levels3,vmin=0,vmax=100,colors='k',linestyles='dotted',linewidths=0.75)
ax2.contour(MUS,SIGMAS,100-Safe_prob.T,levels=levels4,vmin=0,vmax=100,colors='k',linestyles='dashdot',linewidths=0.75)
ax2.contour(MUS,SIGMAS,100-Safe_prob.T,levels=levels5,vmin=0,vmax=100,colors='k',linestyles='dashed',linewidths=0.75)

ax2.plot(tdmean,tdstd,marker='X',color='tab:orange',ms=8)

