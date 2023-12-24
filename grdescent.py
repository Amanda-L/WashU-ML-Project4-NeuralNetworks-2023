# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:03:09 2019

@author: Jerry Xing
"""
import numpy as np
def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-2):
#% function [w]=grdescent(func,w0,stepsize,maxiter,tolerance)
#%
#% INPUT:
#% func function to minimize
#% w0 = initial weight vector 
#% stepsize = initial gradient descent stepsize 
#% tolerance = if norm(gradient)<tolerance, it quits
#%
#% OUTPUTS:
#% 
#% w = final weight vector
#%
    w = w0
    
    loss = 1e-10
    
    for point in range(maxiter):    
            
        # Update w
        # try adaptive step size
        curr_loss = loss
        loss, gradient = func(w)
        
        if loss > curr_loss:
            stepsize *= 0.65
        else:
            stepsize *= 1.03

        w = w - (stepsize * gradient)
        
        # Termination condition
        if np.linalg.norm(gradient) < tolerance:
            break
    return w




