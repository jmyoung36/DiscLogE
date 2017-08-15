#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: jonyoung

calculateCoefficientsTrainExp.pyx

fast calculation of a matrix of coefficients for DiscLogECoeffExp

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_calculateCoefficientsTrainExp (double* array, double* array, double* array, double* array, int n, int dd, int d)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCoefficientsTrainExp(np.ndarray[double, ndim=2, mode="c"] x_logs not None, np.ndarray[double, ndim=2, mode="c"] derivatives not None, np.ndarray[double, ndim=1] x_log_eigenvals not None, np.ndarray[double, ndim=2, mode="c"] coefficients not None):
    """
    calculateCoefficients3 (x_logs, derivatives, coefficients)

    Takes a 4 n by d numpy arrays x_logs, z_logs x_derivatives and z_derivatives
    and an n by 1 arrays x_log_eigenval
    calculate the sum of the elementwise products of the pairs of differences
    for each array (mutliplied by the log eigenvector for the derivatives), 
    writing the results into a n by n array coefficients

    param: array -- a 2-d numpy array of np.float64
    param: value -- a number that will be multiplied by each element in the array

    """
    cdef int n, dd, d

    n, dd = x_logs.shape[0], x_logs.shape[1]
    d = np.sqrt(dd)
    

    out = c_calculateCoefficientsTrainExp (&x_logs[0,0], &derivatives[0,0], &x_log_eigenvals[0], &coefficients[0,0], n, dd, d)

    return out