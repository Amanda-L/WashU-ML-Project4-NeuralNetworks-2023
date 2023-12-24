# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 19:30:29 2019

@author: remus
"""
import numpy as np
def forward_pass(W, xTr, trans_func):
#% function [as,zs]=forward_pass(W,xTr,trans_func)
#%
#% INPUT:
#% W weights (list of numpy array)
#% xTr dxn numpy array (each column is an input vector)
#% trans_func transition function to apply for inner layers
#%
#% OUTPUTS:
#%
#% as = result of forward pass 
#% zs = result of forward pass (zs[0] output layer of the forward pass) 
#%
    n = np.shape(xTr)[1]
    
    ## CHECK!  -JERRY
    
    # First, we add the constant weight
    zzs = [None]*(len(W)+1)
    zzs[-1] = np.vstack((xTr, np.ones([1, n])))
    aas = [None]*(len(W)+1)
    aas[-1] = xTr
    # Do the forward process here
    
    # Each layer takes the output of the previous layer as its input
    for i in range(len(W)-1, -1, -1):
        # Calculate a(l) = W(l)z(l-1) + b(l)
#         print("W:",W[i].shape)
#         print("zzs:",zzs[i+1])
        a_l = np.dot(W[i], zzs[i+1])
        aas[i] = a_l


        # Apply transition function z(l) = g(a(l))
        z_l = trans_func(a_l)
        zzs[i] = np.vstack((z_l,np.ones([1,n])))

  
    # last layer is special, no transition function
    aas[0] = np.dot(W[0], zzs[1])
    zzs[0] = aas[0]    
    
#     print("aas",len(aas))
#     print("zzs",len(zzs))
    return aas, zzs






