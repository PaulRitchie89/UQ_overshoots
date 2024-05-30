# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:36:10 2024

@author: Paul Ritchie

Script to plot Figure 5. Critical freshwater fluxes and width of bistability
region dependence on advective and diffusive timescales.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec

fontsize = 8
rc('font', **{'size' : fontsize})
rc('text', usetex=True)

## Arrays of advective and diffusive timescales (yrs)
ta = np.linspace(57,92,501)
td = np.linspace(70,700,501)
[TA,TD] = np.meshgrid(ta,td)

## Fold bifurcation points for non-dimensionalised salinity (y) and freshwater forcing (p)
A = np.sqrt(1-(3*TA/TD))
y_ = 2/3 - A/3
yp = 2/3 + A/3

p_ = y_*(TA/TD + (1-y_)**2)
pp = yp*(TA/TD + (1-yp)**2)

## Model parameters
alphaT = 1E-4 # Thermal expansion coefficient (/K)
alphaS = 7.6E-4 # Haline contraction coefficient (/psu)
S0 = 35 # Reference salinity (psu)
theta = 25 # Meridional temperature difference (K)
H = 4500 # Mean ocean depth (m)
V = 3.5E16 # Reference ocean volume (m**3)
Area = V/H # Ocean area (m**2)
beta = 365*24*3600 # Seconds in a year (s/yr)
gamma = 1E-6 # Conversion to Sv (Sv s / m**3)
alpha = alphaT*theta*H*Area*gamma/(alphaS*S0)/beta # Freshwater scaling (Sv yrs)

## Fold bifurcation points for dimensionalised freshwater forcing (F)
F_ = alpha*p_/TA
Fp = alpha*pp/TA
## Width of bistability region
Fd = F_-Fp


## Initialise figure
fig = plt.figure(figsize=(7.007874,2.3))

gs=GridSpec(1,6,width_ratios=[0.8,0.25,0.8,0.25,0.8,0.25])
ax=fig.add_subplot(gs[0])
ax1=fig.add_subplot(gs[2])
ax2=fig.add_subplot(gs[4])

ax.set_xlabel('Advective timescale (years)')
ax.set_ylabel('Diffusive timescale (years)')

ax1.set_yticklabels([])
ax1.set_xlabel('Advective timescale (years)')

ax2.set_yticklabels([])
ax2.set_xlabel('Advective timescale (years)')

ax.text(0.87, 0.92, '\\textbf{(a)}', transform=ax.transAxes,size=12,color='w',fontsize=fontsize)
ax1.text(0.87, 0.92, '\\textbf{(b)}', transform=ax1.transAxes,size=12,color='w',fontsize=fontsize)
ax2.text(0.87, 0.92, '\\textbf{(c)}', transform=ax2.transAxes,size=12,color='w',fontsize=fontsize)

## Plotting panel (a)
im = ax.contourf(TA,TD,F_,levels=np.arange(0.15,0.56,step=0.05))
cbar_ax = fig.add_axes([0.275, 0.2, 0.015, 0.75])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.set_label('Upper critical freshwater flux (Sv)')
cbar.set_ticks(np.arange(0.2,0.6,step=0.1))
ax.plot(ta,3*ta,'k',label='$\eta^2 = 3$',lw=0.75)
ax.plot(ta,5*ta,'k--',label='$\eta^2 = 5$',lw=0.75)
ax.plot(ta,7.5*ta,'k:',label='$\eta^2 = 7.5$',lw=0.75)
ax.plot([70, 70],[70, 700],'r--',lw=1)
ax.legend()

## Plotting panel (b)
im = ax1.contourf(TA,TD,Fp,levels=np.arange(0.1,0.56,step=0.05))
cbar_ax = fig.add_axes([0.59, 0.2, 0.015, 0.75])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.set_label('Lower critical freshwater flux (Sv)')
cbar.set_ticks(np.arange(0.1,0.6,step=0.1))
ax1.plot(ta,3*ta,'k',lw=0.75)
ax1.plot(ta,5*ta,'k--',lw=0.75)
ax1.plot(ta,7.5*ta,'k:',lw=0.75)
ax1.plot([70, 70],[70, 700],'r--',lw=1)

## Plotting panel (c)
im = ax2.contourf(TA,TD,Fd,levels=np.concatenate(([0],np.arange(0.01,0.22,step=0.02))))
cbar_ax = fig.add_axes([0.904, 0.2, 0.015, 0.75])
cbar = fig.colorbar(im,cax=cbar_ax,spacing='proportional')
cbar.set_label('Width of freshwater flux bistability\nregion (Sv)')
cbar.set_ticks(np.arange(0,0.25,step=0.1))
ax2.plot(ta,3*ta,'k',lw=0.75)
ax2.plot(ta,5*ta,'k--',lw=0.75)
ax2.plot(ta,7.5*ta,'k:',lw=0.75)
ax2.plot([70, 70],[70, 700],'r--',lw=1)

fig.tight_layout()
