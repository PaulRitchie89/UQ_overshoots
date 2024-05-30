# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:04:14 2024

@author: Paul Ritchie

Script to numerically calculate (bisection method) data for time over lowest
threshold with uncertain tipping threshold location.
Inverse square law data also calculated. 
"""

#import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from matplotlib import rc


def f(y,mu,a0,kappa,delta):

    """
    Normal form saddle-node
    
    Inputs:
        y - state variable
        mu - forcing profile
        delta - threshold location
        kappa - linear restoring force propoortionality constant
    
    Returns:
        dydt - RHS for saddle-node normal form
    """
        
    dydt = a0*(mu - delta + kappa*(y**2))

    return(dydt)


def mu(t,mustart,mushift,muspeed,tend):
    """
    Symmetric return forcing profile
    
    Inputs:
        t - time
        mustart - initial level of forcing
        mushift - peak level of forcing
        muspeed - inversely proportional to rate of forcing
        tend - finishing time of forcing (assumed to start at t = 0)
        
    Returns:
        Symmetric return forcing profile
    """
    return (mustart + (mushift-mustart)/((np.cosh((t-tend/2)/muspeed))**2))    

    
## Time parameters
tend = 600
tspan = [0,tend]
h = 0.01
t = np.arange(tspan[0],tspan[1]+h,h)
nt = len(t)

## System parameters
a0 = 1
kappa = 1
# Probability contour levels for initial distribution of linear restoring force proportionality constant
quantiles = [0.01, 0.1, 0.25, 0.5]
width = 0.3
deltamin = 0
deltas = deltamin+width*np.array(quantiles)

## Forcing parameters
mustart = -2
mushifts = np.linspace(0.006, 0.4, 120)

## Initialise arrays
mutimes = np.zeros((len(deltas),len(mushifts)))
mumax = np.zeros((len(deltas),len(mushifts)))
murs = np.zeros((len(deltas),len(mushifts)))
tesquared = np.zeros((len(deltas),len(mushifts)))

## Number of iterations of bisection method
Nmax = 25

## Loop over all probability contour levels in delta 
for k in range(len(deltas)):
    
    # Determine initial starting position            
    x0 = -np.sqrt(-(mustart-deltas[k])/kappa)
    
    # Loop over all peak forcing levels
    for j in range(len(mushifts)):
        
        mushift = mushifts[j]
        
        # If forcing does not exceed threshold, set time over and inverse rate very large otherwise perform bisection
        if mushift <= deltas[k]:
            murs[k,j], mutimes[k,j] = 2000, 2000
            mumax[k,j] = mushifts[j]
        else:
            
            # Set initial upper & lower bounds for inverse of rate of forcing
            mura = 0        
            murb = 800
            
            N = 1
            
            # Iterate bisection method Nmax times
            while N < Nmax:
                
                # Determine mid-point of upper & lower bounds
                murc = (murb + mura)/2
                
                # Initialise array for state variable and set starting position
                X = np.zeros(len(t)+1)
                X[0] = x0
                
                # Perform Forward Euler
                for i in range(len(t)):
                    X[i+1] = X[i] + h*f(X[i],mu(t[i],mustart,mushift,murc,tend),a0,kappa,deltas[k])
                
                # If system tips set mid point to be the new upper bound otherwise set to be lower bound
                if np.max(X) > 1000:
                    murb = murc
                else:
                    mura = murc
                
                # Increase iteration step by 1
                N = N + 1   
                
            ## Store relevant data
            murs[k,j] = murc
            mutimes[k,j] = h*np.sum(mu(t,mustart,mushift,murc,tend)>0)
            mumax[k,j] = mushifts[j]


        # Calculate theoretical time over lowest threshold. If no overshoot set time over large
        if mumax[k,j] <= deltas[k]:
            tesquared[k,j] = 2000
        else:
            tesquared[k,j] = 4*(mumax[k,j]-deltamin)/((a0**2)*kappa*((mumax[k,j]-deltas[k])**2))

##### SAVE mumax, mutimes and tesquared ######