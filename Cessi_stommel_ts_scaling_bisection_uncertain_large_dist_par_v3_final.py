# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:12:10 2024

@author: Paul Ritchie

Script to numerically calculate (bisection method) data for time over lowest
threshold with uncertain ratio of diffusive to advectove timescales.
Note effectively uncertain diffusive timescale as advective timescale fixed
"""

import numpy as np


def Cessi(y,F,etasquared,decades):

    """
    Stommel 2 box model for AMOC
    
    Inputs:
        y - state variable non-dimensionalised salinity 
        F - freshwater forcing profile
        etasquared - ratio of advective to diffusive timescale
        decades - time rescaling to per decade rather than per year
    
    Returns:
        dydt - RHS for Stommel-Cessi AMOC model
    """
        
    dydt = (F - y*(1+etasquared*((1-y)**2))/etasquared)*decades

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
beta = 365*24*3600 # Seconds in a year (s/yr)
gamma = 1E-6 # Conversion to Sv (Sv s / m**3)
decades = 10 # Time rescaling to decades (yrs/decade)
fscaling = alphaT*theta*H/(alphaS*S0*ta) # Freshwater scaling (m/yr)
x0 = 0 # Initial non-dimensional salinity
# Determine lowest threshold
etasquaredmin = 10 # Ratio of timescales corresponding to lowest threshold
Amin = np.sqrt(1/9 - 1/(3*etasquaredmin))
Fminusmin = 2/27 + 2/(3*etasquaredmin) + Amin*(1/3-1/etasquaredmin-Amin**2)
# Contour levels for ratio of timescales
quantiles = [0.01,0.1,0.25,0.5]
width = 7
etasquareds = etasquaredmin - width*np.array(quantiles)


## Freshwater forcing parameters
Fstart = 0
Fstab = 1/fscaling
r = 0.01
# Initialise array of short term transition timescales  
mu0s_int = np.concatenate((np.linspace(0.03,0.00001,31),[0]))
mu0s = np.concatenate((mu0s_int,np.geomspace(-0.00001,-5.2,101)))

## Initialise arrays
Ftimes = np.zeros((len(etasquareds),len(mu0s)))
Fmax = np.zeros((len(etasquareds),len(mu0s)))
Frs = np.zeros((len(etasquareds),len(mu0s)))


## Number of iterations of bisection method
Nmax = 30

## Loop over all probability contour levels in etasquared 
for k in range(len(etasquareds)):    
    
    # Calculate fold points in both freshwater and non-dimenisonalised salinity
    A = np.sqrt(1/9 - 1/(3*etasquareds[k]))
    
    xplus = 2/3 + A
    xminus = 2/3 - A

    Fplus = 2/27 + 2/(3*etasquareds[k]) - A*(1/3-1/etasquareds[k]-A**2)
    Fminus = 2/27 + 2/(3*etasquareds[k]) + A*(1/3-1/etasquareds[k]-A**2)
    
    # Loop over all short term transition timescales
    for j in range(len(mu0s)):        
    
        # Set initial upper & lower bounds for long term transition timescale
        mu1a = -0.00005        
        mu1b = 12
        
        N = 1
        
        # Iterate bisection method Nmax times
        while N < Nmax:
            
            # Determine mid-point of upper & lower bounds
            mu1c = (mu1b + mu1a)/2
            
            # Initialise array for state variable and set starting position
            X = np.zeros(len(t)+1)
            X[0] = x0
            
            # Perform Forward Euler
            for i in range(len(t)):
                X[i+1] = X[i] + h*Cessi(X[i],F(t[i],Fstart,Fstab,r,mu0s[j],mu1c),etasquareds[k],decades)
            
            # If system tips set mid point to be the new lower bound otherwise set to be upper bound
            if np.max(X) > xplus:
                mu1a = mu1c
            else:
                mu1b = mu1c
            
            # Increase iteration step by 1
            N = N + 1   
            
        ## Store relevant data
        Frs[k,j] = mu1c
        if mu1c < -4.9977E-5:
            Ftimes[k,j] = np.nan
        else:
            Ftimes[k,j] = h*np.sum(F(t,Fstart,Fstab,r,mu0s[j],mu1c)>Fminusmin)
        Fmax[k,j] = np.max(F(t,Fstart,Fstab,r,mu0s[j],mu1c))
        
        ## If critical overshoot is sufficiently large break loop and consider next etasquared value
        if Fmax[k,j] > 1.75/fscaling:
            Ftimes[k,j+1:] = np.nan
            break

##### SAVE Fmax and Ftimes ######        
