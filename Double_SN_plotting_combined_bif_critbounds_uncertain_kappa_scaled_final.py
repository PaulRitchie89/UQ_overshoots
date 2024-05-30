# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:50:30 2024

@author: Paul Ritchie

Script to plot Figure 3.  Probabilistic overshoots given uncertainty in strength
of linear restoring force.
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
tend = 80
tspan = [0,tend]
h = 0.01
t = np.arange(tspan[0],tspan[1]+h,h)
nt = len(t)

## System parameters
a0 = 1
delta = 2
x0 = 2.5
# Consider 2 different linear restoring force proportionality constants for illustration in Figure 3b
kappas = [1,2] 


## Forcing parameters
mustart = 0
mushift = 0.1 + delta
r = 0.1

## Initialising figure
fig = plt.figure(figsize=(7.007874,3.2849409375))
gs = gridspec.GridSpec(3, 2)
ax0 = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs[1:,0])
ax2 = plt.subplot(gs[:,1])

ax0.set_xlim(0,tend)
ax1.set_xlim(-0.15,2.12)
ax1.set_ylim(0,2.6)
ax2.set_xlim(0,0.15)
ax2.set_ylim(2,10)

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
ax2.set_xlabel('Peak overshoot distance')
ax2.set_ylabel('Time over threshold')

fig.tight_layout()

ax0.text(0.92, 0.92, '\\textbf{(a)}', transform=ax0.transAxes,size=12,fontsize=fontsize,bbox=dict(facecolor='w', edgecolor='none',pad=2))
ax1.text(0.92, 0.92, '\\textbf{(b)}', transform=ax1.transAxes,size=12,fontsize=fontsize)
ax2.text(0.92, 0.975, '\\textbf{(c)}', transform=ax2.transAxes,size=12,fontsize=fontsize)


# Plotting differentials between scenarios
alphas = [1,0.5]
labels = ['Weak restoring force \n prevents tipping', 'Strong restoring force \n promotes tipping']
lws = [0.75,2]
lss = [':','-.','--','-','--',':']
cols = ['tab:orange','r']
xarrows = [1.44,0.3]

## Plotting panel (a)
ax0.plot(tspan,[0+delta,0+delta],c='r',ls=':',lw=0.75)
ax0.plot(t,mu(t,mustart,mushift,r,tend),c='b',lw=1)

## Initialise array of forcing values for equilibria
mueq = np.linspace(-0.5,2.5,10001)

## Loop over different linear restoring force proportionality constants
for k in range(len(kappas)):
    
    # Ensure state variable starting position is the same for each scenario 
    epsilon = x0 - np.sqrt(delta/kappas[k])
    
    # Calculate equilibria branches and plot
    xeqplus = epsilon + np.sqrt((-mueq+delta)/kappas[k])
    xeqminus = epsilon - np.sqrt((-mueq+delta)/kappas[k])
    ax1.plot(mueq,xeqminus,'k--',alpha=alphas[k],lw=lws[k])
    ax1.plot(mueq,xeqplus,'k',alpha=alphas[k],lw=lws[k])

    # Initialise array for state variable and set starting position
    X = np.zeros(len(t)+1)
    X[0] = x0
    
    # Perform Forward Euler
    for i in range(len(t)):
        X[i+1] = X[i] + h*f(X[i],mu(t[i],mustart,mushift,r,tend),a0,kappas[k],delta,epsilon)

    
    # Plotting panel (b)
    ax1.plot(mu(t,mustart,mushift,r,tend),X[:-1],c='b',alpha=alphas[k],label=labels[k],lw=lws[k])
    ax1.plot(mu(t[0],mustart,mushift,r,tend),X[0],'.',c='b',ms=8,alpha=alphas[k])

    ax1.plot(0+delta,0+epsilon,'.',c='r',ms=8,alpha=alphas[k])
    ax1.plot([0+delta,0+delta],[-0.7,0+epsilon],c='r',ls=':',alpha=alphas[k],lw=lws[k])
    
    idx = np.nanargmin(np.abs(mu(t,mustart,mushift,r,tend) - 1.7)+np.abs(X[:-1]-xarrows[k]))
    arrow0 = mu(t[idx+1],mustart,mushift,r,tend), X[idx+1]
    arrow1 = mu(t[idx],mustart,mushift,r,tend), X[idx]
    ax1.annotate('',xytext=(arrow1),xy=(arrow0),arrowprops=dict(arrowstyle="simple", color='b',alpha=alphas[k],lw=lws[k],edgecolor='none'),size=12)


ax1.legend(frameon=False)

## Numerically calculate time spent over threshold for forcing profile used in (a)
timeover = h*np.sum(mu(t,mustart,mushift,r,tend)>delta)

## Load in data from bisection method used to calculate critical boundaries
## tesquared - theoretical time over, mutimes - numerical time over
mat = loadmat('SN_overshoots_uncertain_kappa_data_v3.mat')
mumax = mat['mumax'][:]
tesquared = mat['tesquared'][:]
mutimes = mat['mutimes'][:]

## Plotting panel (c) - theory in colour and numerical as black contours
ax2.fill_between(mumax[0,:],np.sqrt(tesquared[0,:]),np.sqrt(tesquared[1,:]),color=[136/255,46/255,114/255],alpha=0.2,edgecolor='none')
ax2.fill_between(mumax[0,:],np.sqrt(tesquared[1,:]),np.sqrt(tesquared[2,:]),color=[136/255,46/255,114/255],alpha=0.4,edgecolor='none')
ax2.fill_between(mumax[0,:],np.sqrt(tesquared[2,:]),np.sqrt(tesquared[3,:]),color=[136/255,46/255,114/255],alpha=0.6,edgecolor='none')
for k in range(4):
    ax2.plot(mumax[k,:], mutimes[k,:],lw=0.75,c='k',ls=lss[k])  


# Add location of profile given in panel (a)
ax2.plot(mushift-delta,timeover,'X',c='b',ms=7)
