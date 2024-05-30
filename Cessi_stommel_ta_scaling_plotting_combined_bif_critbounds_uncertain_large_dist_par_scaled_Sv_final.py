# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:11:50 2024

@author: Paul Ritchie

Script to plot Figure 6. Probabilistic overshoots given uncertainty in
diffusive timescale of AMOC Stommel-Cessi model.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from scipy.io import loadmat

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

def Q(y,etasquared,ta,beta,V,gamma):
    """
    Transport function Q
    
    Inputs:
        y - state variable non-dimensionalised salinity
        etasquared - ratio of diffusive to advective timescale
        ta - advective timescale (yrs)
        beta - seconds in a year (s/yr)
        V - reference ocean volume (m**3)
        gamma - conversion to Sv (Sv s / m**3)
        
    Returns:
        Flow strength of the AMOC (Sv)
    """
    return (V*gamma*(1+etasquared*((1-y)**2))/etasquared/ta/beta)


## Time parameters
tend = 50
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
x0 = 0 # Initial non-dimensional salinity
Qstart = Q(0,7.5,ta,beta,V,gamma) # Initial AMOC strength with default ratio of timescales
# Consider 2 different diffusive timescales for illustration in Figure 6b
tds = [455,700]
etasquareds = np.array(tds)/ta
etasquaredsmin = np.max(etasquareds)
# Contour levels for ratio of timescales
quantiles = [0.01,0.1,0.25,0.5]
width = 7
etasquareds2 = etasquaredsmin - width*np.array(quantiles)

## Non-diemnsionalised forcing parameters
Fstart = 0
Fstab = 1*Area*gamma/beta/fscaling
r = 0.01
mu0 = -0.05
mu1 = 0.0057
mushift = np.max(F(t,Fstart,Fstab,r,mu0,mu1))

## Calculating fold bifurcation points
Amin_bnd = np.sqrt(1/9 - 1/(3*etasquaredsmin))
muminusmin_bnd = 2/27 + 2/(3*etasquaredsmin) + Amin_bnd*(1/3-1/etasquaredsmin-Amin_bnd**2)

Amin = np.sqrt(1/9 - 1/(3*etasquareds[0]))
muminusmin = 2/27 + 2/(3*etasquareds[0]) + Amin*(1/3-1/etasquareds[0]-Amin**2)
muplusmin = 2/27 + 2/(3*etasquareds[0]) - Amin*(1/3-1/etasquareds[0]-Amin**2)
Amax = np.sqrt(1/9 - 1/(3*etasquareds[1]))
muminusmax = 2/27 + 2/(3*etasquareds[1]) + Amax*(1/3-1/etasquareds[1]-Amax**2)
muplusmax = 2/27 + 2/(3*etasquareds[1]) - Amax*(1/3-1/etasquareds[1]-Amax**2)

xminus = [2/3 - Amin,2/3 - Amax]
xplus = [2/3 + Amin,2/3 + Amax]
muminus = [muminusmin,muminusmax]
muplus = [muplusmin,muplusmax]


## Initialising figure
fig = plt.figure(figsize=(7.007874,3.2849409375))
gs = gridspec.GridSpec(3, 2)
ax0 = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs[1:,0])
ax2 = plt.subplot(gs[:,1])

ax0.set_xlim(0,tend*ta)
ax1.set_xlim(-0.1*Area/beta/1E6,1.5*Area/beta/1E6)
ax1.set_ylim(0,20)
ax2.set_xlim(1.1*Area/beta/1E6,1.5*Area/beta/1E6)
ax2.set_ylim(0,5000)

ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax0.set_xlabel('Time (years)')
ax0.set_ylabel('Freshwater flux \n (Sv)')
ax1.set_xlabel('Freshwater flux (Sv)')
ax1.set_ylabel('AMOC strength (Sv)')
ax2.set_xlabel('Peak freshwater flux (Sv)')
ax2.set_ylabel('Time over lowest threshold (years)')

fig.tight_layout()

ax0.text(0.92, 0.92, '\\textbf{(a)}', transform=ax0.transAxes,size=12,fontsize=fontsize)
ax1.text(0.92, 0.92, '\\textbf{(b)}', transform=ax1.transAxes,size=12,fontsize=fontsize)
ax2.text(0.92, 0.975, '\\textbf{(c)}', transform=ax2.transAxes,size=12,fontsize=fontsize)

# Plotting differentials between scenarios
alphas = [1,0.5]
labels = ['Small diffusive timescale \n prevents tipping', 'Large diffusive timescale \n promotes tipping']
lws = [0.75,2]
lss = [':','-.','--','-','--',':']
cols = ['tab:orange','r']
xarrows = [1,0.65]

## Plotting panel (a)
ax0.plot([value * ta for value in tspan],[fscaling*muminusmin,fscaling*muminusmin],ls=':',c=cols[0],alpha=alphas[0],lw=lws[0])
ax0.plot([value * ta for value in tspan],[fscaling*muminusmax,fscaling*muminusmax],ls=':',c=cols[1],alpha=alphas[1],lw=lws[1])
ax0.plot(t*ta,fscaling*F(t,Fstart,Fstab,r,mu0,mu1),'b',lw=1)

## Initialise array of non-dimensionalised salinity for equilibria
xeq = np.linspace(-0.5,3,10001)

## Loop over different ratios of timescales
for k in range(len(etasquareds)):
    
    # Adjust volume so that the state variable starting position is the same for each scenario 
    V = Qstart*ta*etasquareds[k]*beta/(1+etasquareds[k])/gamma
    
    # Calculate equilibria branches and plot
    Feq = xeq*(1+etasquareds[k]*((1-xeq)**2))/etasquareds[k]
    ax1.plot(fscaling*Feq[(xeq>xminus[k])&(xeq<xplus[k])],Q(xeq[(xeq>xminus[k])&(xeq<xplus[k])],etasquareds[k],ta,beta,V,gamma),'k--',alpha=alphas[k],lw=lws[k])
    ax1.plot(fscaling*Feq[xeq<xminus[k]],Q(xeq[xeq<xminus[k]],etasquareds[k],ta,beta,V,gamma),'k',alpha=alphas[k],lw=lws[k])
    ax1.plot(fscaling*Feq[xeq>xplus[k]],Q(xeq[xeq>xplus[k]],etasquareds[k],ta,beta,V,gamma),'k',alpha=alphas[k],lw=lws[k])
        
    # Initialise array for state variable and set starting position
    X = np.zeros(len(t)+1)
    X[0] = x0
    
    # Perform Forward Euler
    for i in range(len(t)):
        X[i+1] = X[i] + h*Cessi(X[i],F(t[i],Fstart,Fstab,r,mu0,mu1),etasquareds[k])

    
    # Plotting panel (b)
    ax1.plot(fscaling*F(t,Fstart,Fstab,r,mu0,mu1),Q(X[:-1],etasquareds[k],ta,beta,V,gamma),'b',alpha=alphas[k],label=labels[k],lw=lws[k])
    ax1.plot(fscaling*F(t[0],Fstart,Fstab,r,mu0,mu1),Q(X[0],etasquareds[k],ta,beta,V,gamma),'b.',ms=8,alpha=alphas[k])
    ax1.plot(fscaling*F(t[-1],Fstart,Fstab,r,mu0,mu1),Q(X[-1],etasquareds[k],ta,beta,V,gamma),'b.',ms=8,alpha=alphas[k])

    ax1.plot(fscaling*muminus[k],Q(xminus[k],etasquareds[k],ta,beta,V,gamma),'.',c=cols[k],ms=8,alpha=alphas[k])
    ax1.plot([fscaling*muminus[k],fscaling*muminus[k]],[0,Q(xminus[k],etasquareds[k],ta,beta,V,gamma)],':',c=cols[k],alpha=alphas[k],lw=lws[k])
    
    idx = np.nanargmin(np.abs(F(t,Fstart,Fstab,r,mu0,mu1) - 1.45)+np.abs(X[:-1]-xarrows[k]))
    arrow0 = fscaling*F(t[idx+1],Fstart,Fstab,r,mu0,mu1), Q(X[idx+1],etasquareds[k],ta,beta,V,gamma)
    arrow1 = fscaling*F(t[idx],Fstart,Fstab,r,mu0,mu1), Q(X[idx],etasquareds[k],ta,beta,V,gamma)
    ax1.annotate('',xytext=(arrow1),xy=(arrow0),arrowprops=dict(arrowstyle="simple", color='b',alpha=alphas[k],lw=lws[k],edgecolor='none'),size=12)

ax1.legend(frameon=False,loc='lower left', bbox_to_anchor=(0.02, 0.2))

## Numerically calculate time spent over threshold for forcing profile used in (a)
timeover = h*np.sum(F(t,Fstart,Fstab,r,mu0,mu1)>muminusmin_bnd)


## Load in data from bisection method used to calculate critical boundaries
decades = 10
decades_string = '_decades'
mat = loadmat('AMOC_Cessi_tascaling_small_overshoots_Fstart0_Fstab1_r001_Ftimesworsteta'+decades_string+'_large_uncertain_etasquared_data_v2.mat')
mumax = mat['Fmax'][:]
mutimes = decades*mat['Ftimes'][:]


## Calculate theoretical time over from inverse square law
Fmax2 = np.linspace(1*Area*gamma/beta/fscaling,2*Area*gamma/beta/fscaling,501)
tesquared = np.zeros((len(etasquareds2),len(Fmax2)))
for k in range(len(etasquareds2)):
    for j in range(len(Fmax2)):
        A = np.sqrt(1/9 - 1/(3*etasquareds2[k]))
        kappa = 3*A 
        Fminus = 2/27 + 2/(3*etasquareds2[k]) + A*(1/3-1/etasquareds2[k]-A**2)
        # If no overshoot set time over very large else calculate using the inverse square law
        if Fmax2[j] <= Fminus:
            tesquared[k,j] = 100000
        else:    
            tesquared[k,j] = 4*(Fmax2[j]-muminusmin_bnd)/(kappa*((Fmax2[j]-Fminus)**2))


## Plotting panel (c) - theory in colour and numerical as black contours
ax2.fill_between(fscaling*Fmax2,ta*np.sqrt(tesquared[0,:]),ta*np.sqrt(tesquared[1,:]),color=[136/255,46/255,114/255],alpha=0.2,edgecolor='none')
ax2.fill_between(fscaling*Fmax2,ta*np.sqrt(tesquared[1,:]),ta*np.sqrt(tesquared[2,:]),color=[136/255,46/255,114/255],alpha=0.4,edgecolor='none')
ax2.fill_between(fscaling*Fmax2,ta*np.sqrt(tesquared[2,:]),ta*np.sqrt(tesquared[3,:]),color=[136/255,46/255,114/255],alpha=0.6,edgecolor='none')
for k in range(len(quantiles)):
    ax2.plot(fscaling*mumax[k,:], ta*mutimes[k,:],lw=0.75,c='k',ls=lss[k])  


# Add location of profile given in panel (a)
ax2.plot(fscaling*mushift,ta*timeover,'bX',ms=7)
