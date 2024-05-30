# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:05:21 2024

@author: Paul Ritchie

Script to plot Figure 1. Probabilistic overshoots given uncertainty in
location of tipping threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from scipy.io import loadmat

fontsize = 8
rc('font', **{'size' : fontsize})
rc('text', usetex=True)

def f(y,mu,a0,kappa,delta,epsilon):

    """
    Normal form saddle-node
    
    Inputs:
        y - state variable
        mu - forcing profile
        delta - threshold location
        kappa - linear restoring force propoortionality constant
        epsilon - ensures same starting position y0 at t = 0
    
    Returns:
        dydt - RHS for saddle-node normal form
    """
        
    dydt = a0*(-mu + delta - kappa*((y - epsilon)**2))

    return(dydt)
    

    
def mu(t,mustart,mushift,r,tend):
    """
    Symmetric return forcing profile
    
    Inputs:
        t - time
        mustart - initial level of forcing
        mushift - peak level of forcing
        r - proportional to rate of forcing
        tend - finishing time of forcing (assumed to start at t = 0)
        
    Returns:
        Symmetric return forcing profile
    """
    return (mustart + (mushift-mustart)/((np.cosh(r*(t-tend/2)))**2))


## Time parameters
tend = 50
tspan = [0,tend]
h = 0.01
t = np.arange(tspan[0],tspan[1]+h,h)
nt = len(t)

## System parameters
a0 = 1
kappa = 1
x0 = 2.5
# Consider 2 different threshold locations for illustration in Figure 1b
deltas = [2.3,2] 


## Forcing parameters
mustart = 0
mushift = 2.35
r = 0.24


## Initialising figure
fig = plt.figure(figsize=(7.007874,3.2849409375))
gs = gridspec.GridSpec(3, 2)
ax0 = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs[1:,0])
ax2 = plt.subplot(gs[:,1])

ax0.set_xlim(0,tend)
ax1.set_xlim(-0.15,2.4)
ax1.set_ylim(0,2.6)
ax2.set_xlim(2,2.4)
ax2.set_ylim(0,40)

ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax0.set_xlabel('Time')
ax0.set_ylabel('External forcing')
ax1.set_xlabel('External forcing')
ax1.set_ylabel('System state')
ax2.set_xlabel('Peak external forcing')
ax2.set_ylabel('Time over lowest threshold')

fig.tight_layout()

ax0.text(0.92, 0.92, '\\textbf{(a)}', transform=ax0.transAxes,size=12,fontsize=fontsize,bbox=dict(facecolor='w', edgecolor='none',pad=2))
ax1.text(0.92, 0.92, '\\textbf{(b)}', transform=ax1.transAxes,size=12,fontsize=fontsize)
ax2.text(0.92, 0.975, '\\textbf{(c)}', transform=ax2.transAxes,size=12,fontsize=fontsize)

# Plotting differentials between scenarios
alphas = [1,0.5]
labels = ['High threshold prevents tipping', 'Low threshold promotes tipping']
lws = [0.75,2]
lss = [':','-.','--','-','--',':']
cols = ['tab:orange','r']
xarrows = [1.09,0.58]

## Plotting panel (a)
ax0.plot(tspan,[deltas[0],deltas[0]],c=cols[0],alpha=alphas[0],lw=lws[0],ls=':')
ax0.plot(tspan,[deltas[1],deltas[1]],c=cols[1],alpha=alphas[1],lw=lws[1],ls=':')
ax0.plot(t,mu(t,mustart,mushift,r,tend),c='b',lw=1)


## Initialise array of forcing values for equilibria
mueq = np.linspace(-0.5,2.5,10001)

## Loop over different threshold locations
for k in range(len(deltas)):
    
    # Ensure state variable starting position is the same for each scenario 
    epsilon = x0 - np.sqrt(deltas[k]/kappa)
    
    # Calculate equilibria branches and plot
    xeqplus = epsilon + np.sqrt((-mueq+deltas[k])/kappa)
    xeqminus = epsilon - np.sqrt((-mueq+deltas[k])/kappa)
    ax1.plot(mueq,xeqminus,'k--',alpha=alphas[k],lw=lws[k])
    ax1.plot(mueq,xeqplus,'k',alpha=alphas[k],lw=lws[k])
    
    # Initialise array for state variable and set starting position
    X = np.zeros(len(t)+1)
    X[0] = x0
    
    # Perform Forward Euler
    for i in range(len(t)):
        X[i+1] = X[i] + h*f(X[i],mu(t[i],mustart,mushift,r,tend),a0,kappa,deltas[k],epsilon)

    
    ## Plotting panel (b)
    ax1.plot(mu(t,mustart,mushift,r,tend),X[:-1],c='b',alpha=alphas[k],label=labels[k],lw=lws[k])
    ax1.plot(mu(t[0],mustart,mushift,r,tend),X[0],'.',c='b',ms=8,alpha=alphas[k])

    ax1.plot(0+deltas[k],0+epsilon,'.',c=cols[k],ms=8,alpha=alphas[k])
    ax1.plot([0+deltas[k],0+deltas[k]],[-0.7,0+epsilon],':',c=cols[k],alpha=alphas[k],lw=lws[k])
    
    idx = np.nanargmin(np.abs(mu(t,mustart,mushift,r,tend) - 1.7)+np.abs(X[:-1]-xarrows[k]))
    arrow0 = mu(t[idx+1],mustart,mushift,r,tend), X[idx+1]
    arrow1 = mu(t[idx],mustart,mushift,r,tend), X[idx]
    ax1.annotate('',xytext=(arrow1),xy=(arrow0),arrowprops=dict(arrowstyle="simple", color='b',alpha=alphas[k],lw=lws[k],edgecolor='none'),size=12)

ax1.legend(frameon=False)

## Numerically calculate time spent over lowest threshold for forcing profile used in (a)
timeover = h*np.sum(mu(t,mustart,mushift,r,tend)>np.min(deltas))

## Load in data from bisection method used to calculate critical boundaries
## tesquared - theoretical time over, mutimes - numerical time over
mat = loadmat('SN_overshoots_uncertain_thresh_data_v3.mat')
mumax = mat['mumax'][:]
tesquared = mat['tesquared'][:]
mutimes = mat['mutimes'][:]

## Plotting panel (c) - theory in colour and numerical as black contours
ax2.fill_between(mumax[0,:]+np.min(deltas),np.sqrt(tesquared[0,:]),np.sqrt(tesquared[1,:]),color=[136/255,46/255,114/255],alpha=0.2,edgecolor='none')
ax2.fill_between(mumax[0,:]+np.min(deltas),np.sqrt(tesquared[1,:]),np.sqrt(tesquared[2,:]),color=[136/255,46/255,114/255],alpha=0.4,edgecolor='none')
ax2.fill_between(mumax[0,:]+np.min(deltas),np.sqrt(tesquared[2,:]),np.sqrt(tesquared[3,:]),color=[136/255,46/255,114/255],alpha=0.6,edgecolor='none')
for k in range(4):
    ax2.plot(mumax[k,:]+np.min(deltas), mutimes[k,:],lw=0.75,c='k',ls=lss[k])  
   
# Add location of profile given in panel (a)
ax2.plot(mushift,timeover,'X',c='b',ms=7)
