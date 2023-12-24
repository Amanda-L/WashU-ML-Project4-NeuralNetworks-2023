# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:07:53 2019

@author: Jerry Xing
"""
import numpy as np
def backprop(W, aas,zzs, yTr,  trans_func_der):
#% function [gradient] = backprop(W, aas, zzs, yTr,  der_trans_func)
#%
#% INPUT:
#% W weights (list of ndarray)
#% aas output of forward pass (list of ndarray)
#% zzs output of forward pass (list of ndarray)
#% yTr 1xn ndarray (each entry is a label)
#% der_trans_func derivative of transition function to apply for inner layers
#%
#% OUTPUTS:
#% 
#% gradient = the gradient at w as a list of ndarries
#%

    n = np.shape(yTr)[1]
    
    # this is the δ notation in notes
    delta = zzs[0] - yTr 
#     delta = [None] * len(aas)
#     delta = (zzs[0] - yTr )* trans_func_der(aas[0])
    
    
    # compute gradient with back-prop
    gradient = [None] * len(W)
    for i in range(len(W)):
        # gradient(L) = δ(L)*z(L-1)
        gradient[i] = np.dot(delta, zzs[i+1].T)/n
        
        # δ(L-1) = g'(a(L-1)) ⊙ W(L)* δ(L)
        delta = np.multiply(trans_func_der(aas[i+1]), np.dot(W[i][:,:-1].T, delta))

    return gradient 


    
#     gamma[0] = delta * trans_func_der(zzs[0]) #computing loss in output layer
#     gamma[1] = np.dot(zzs[1] * W[0].T, gamma[0].T)
#     alpha = .001
#     W[0] = W[0] - np.dot(alpha*gamma[0],zzs[1].T)
#     for i in range(len(W)):
#         gradient[i] = np.dot(gamma[i],zzs[i-1].T)
#         gamma[i-1] = np.dot(zzs[i-1]*W[i].T,gamma[i])
#         W[i]  = W[i] - alpha*gradient[i]



#     return gradient 