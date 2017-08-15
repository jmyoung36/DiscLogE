#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: jonyoung

calculateCoefficientsTrain.pyx

fast calculation of a matrix of coefficients for DiscLogECoeff

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_calculateCoefficientsTrain (double* array, double* array, double* array, int n, int dd, int d)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCoefficientsTrain(np.ndarray[double, ndim=2, mode="c"] x_logs not None, np.ndarray[double, ndim=2, mode="c"] derivatives not None, np.ndarray[double, ndim=2, mode="c"] coefficients not None):
    """
    calculateCoefficients3 (x_logs, derivatives, coefficients)

    Takes a 2 n by d numpy arrays x_logs and derivatives
    calculate the sum of the elementwise products of the pairs of differences
    for each array, writing the results into a n by n array coefficients

    param: array -- a 2-d numpy array of np.float64
    param: value -- a number that will be multiplied by each element in the array

    """
    cdef int n, dd, d

    n, dd = x_logs.shape[0], x_logs.shape[1]
    d = np.sqrt(dd)
    

    out = c_calculateCoefficientsTrain (&x_logs[0,0], &derivatives[0,0], &coefficients[0,0], n, dd, d)

    return out