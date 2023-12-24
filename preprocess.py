# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:05:03 2019

@author: Jerry Xing
"""
import numpy as np
def preprocess(xTr,xTe):
# function [xTr,xTe,u,m]=preprocess(xTr,xTe);
#
# Preproces the data to make the training features have zero-mean and
# standard-deviation 1
# input:
# xTr - raw training data as d by n_train numpy ndarray 
# xTe - raw test data as d by n_test numpy ndarray
    
# output:
# xTr - pre-processed training data 
# xTe - pre-processed testing data
#
# u,m - any other data should be pre-processed by x-> u*(x-m)
#       where u is d by d ndnumpy array and m is d by 1 numpy ndarray
    
    d, _ = np.shape(xTr)
#     m = np.zeros((d,1))
#     u = np.zeros((d,d))    
    ## << Remove 2 lines above and insert your solution here
    
#     mean and std of a row (i.e, of the features)
    x_mean = np.mean(xTr, axis=1).reshape(-1, 1)
    x_std = np.std(xTr, axis=1).reshape(-1, 1)



    # transformation matrix u = adiagonal matrix with entries 1/std

    u = np.diag(1/x_std.flatten())  #(13x13)
    # print("U shape:",u.shape)
    # print("xTr Shape:",xTr.shape)
    # print("xMean shape:",x_mean.shape)
    # print("subtract shape:", (xTr - x_mean).shape)
    # x-> u*(x-m)
    subtract = (xTr - x_mean)   #(13x305)
    subtractTE = (xTe - x_mean) #(13xnumber of samples)
    xTr = u.dot(subtract) #(13,305)
    xTe = u.dot(subtractTE) #(13xnumber of samples)
    m = x_mean #(dx1) so (13x1)

    return xTr, xTe, u, m

