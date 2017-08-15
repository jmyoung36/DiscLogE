#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: jonyoung

calculateCoefficientsCrossPointer.pyx

fast calculation of a matrix of coefficients for DiscLogEExp

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_calculateCoefficientsCrossExpPointer (double* array, double* array, double* array, double* array, double* array, double* array, double* array, int n_x, int n_z, int dd, int d)

@cython.boundscheck(False)
@cython.wraparound(False)
def calculateCoefficientsCrossExpPointer(np.ndarray[double, ndim=2, mode="c"] x_logs not None, np.ndarray[double, ndim=2, mode="c"] z_logs not None, np.ndarray[double, ndim=2, mode="c"] x_derivatives not None, np.ndarray[double, ndim=2, mode="c"] z_derivatives not None, np.ndarray[double, ndim=1] x_log_eigenvals not None, np.ndarray[double, ndim=1] z_log_eigenvals not None, np.ndarray[double, ndim=2, mode="c"] coefficients not None):
    """
    calculateCoefficientsCrossExpPointer (x_logs, derivatives, coefficients)

    Takes a 4 n by d numpy arrays x_logs, z_logs x_derivatives and z_derivatives
    and 2 n by 1 arrays x_log_eigenvals and z_log_eigenvals
    calculate the sum of the elementwise products of the pairs of differences
    for each array (mutliplied by the log eigenvector for the derivatives), 
    writing the results into a n by n array coefficients

    """
    cdef int n, dd, d

    n_x, dd = x_logs.shape[0], x_logs.shape[1]
    n_z = z_logs.shape[0]
    d = np.sqrt(dd)
    

    out = c_calculateCoefficientsCrossExpPointer (&x_logs[0,0], &z_logs[0,0], &x_derivatives[0,0], &z_derivatives[0,0], &x_log_eigenvals[0], &z_log_eigenvals[0], &coefficients[0,0], n_x, n_z, dd, d)

    return out