#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: jonyoung

calculateCoefficientsCross.pyx

fast calculation of a matrix of coefficients for DiscLogECoeff

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_calculateCoefficientsCross (double* array, double* array, double* array, double* array, double* array, int n_x, int n_z, int dd, int d)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCoefficientsCross(np.ndarray[double, ndim=2, mode="c"] x_logs not None, np.ndarray[double, ndim=2, mode="c"] z_logs not None, np.ndarray[double, ndim=2, mode="c"] x_derivatives not None, np.ndarray[double, ndim=2, mode="c"] z_derivatives not None, np.ndarray[double, ndim=2, mode="c"] coefficients not None):
    """
    calculateCoefficientsCross (x_logs, derivatives, coefficients)

    Takes a 4 n by d numpy arrays x_logs, z_logs x_derivatives and z_derivatives
    calculate the sum of the elementwise products of the pairs of differences
    for each array, writing the results into a n by n array coefficients

    """
    cdef int n, dd, d

    n_x, dd = x_logs.shape[0], x_logs.shape[1]
    n_z = z_logs.shape[0]
    d = np.sqrt(dd)
    

    out = c_calculateCoefficientsCross (&x_logs[0,0], &z_logs[0,0], &x_derivatives[0,0], &z_derivatives[0,0], &coefficients[0,0], n_x, n_z, dd, d)

    return out