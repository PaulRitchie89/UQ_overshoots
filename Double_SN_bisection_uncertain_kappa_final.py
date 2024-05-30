# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:03:19 2024

@author: Paul Ritchie

Script to numerically calculate (bisection method) data for time over tipping
threshold with uncertain linear restoring rate proportionality constant.
Inverse square law data also calculated. 
"""

import numpy as np


def f(y,mu,a0,kappa):

    """
    Normal form saddle-node
    
    Inputs:
        y - state variable
        mu - forcing profile
        kappa - linear restoring force propoortionality constant
    
    Returns:
        dydt - RHS for saddle-node normal form
    """
        
    dydt = a0*(mu + kappa*(y**2))

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
tend = 4000
tspan = [0,tend]
h = 0.02
t = np.arange(tspan[0],tspan[1]+h,h)
nt = len(t)

## System parameters
a0 = 1
# Probability contour levels for initial distribution of linear restoring force proportionality constant
quantiles = [0.99, 0.9, 0.75, 0.5]
width = 3
kappamin = 0.25
kappas = kappamin+width*np.array(quantiles)

## Forcing parameters
mustart = -2
mushifts = np.geomspace(0.002, 0.15, 30)

## Initialise arrays
mutimes = np.zeros((len(kappas),len(mushifts)))
mumax = np.zeros((len(kappas),len(mushifts)))
murs = np.zeros((len(kappas),len(mushifts)))
tesquared = np.zeros((len(kappas),len(mushifts)))


## Number of iterations of bisection method
Nmax = 25

## Loop over all probability contour levels in kappa 
for k in range(len(kappas)):    
    
    # Determine initial starting position           
    x0 = -np.sqrt(-kappas[k]*mustart)
    
    # Loop over all overshoot distances
    for j in range(len(mushifts)):
        
        mushift = mushifts[j]
    
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
                X[i+1] = X[i] + h*f(X[i],mu(t[i],mustart,mushift,murc,tend),a0,kappas[k])
            
            # If system tips set mid point to be the new upper bound otherwise set to be lower bound 
            if np.max(X) > 10:
                murb = murc
            else:
                mura = murc
            
            # Increase iteration step by 1   
            N = N + 1   
            
        ## Store relevant data
        murs[k,j] = murc
        mutimes[k,j] = h*np.sum(mu(t,mustart,mushift,murc,tend)>0)
        mumax[k,j] = mushifts[j]
        
    tesquared[k,:] = 4/((a0**2)*kappas[k]*mumax[k,:])
        
##### SAVE mumax, mutimes and tesquared ######