#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:53:40 2017

@author: jonyoung
"""

#from distutils.core import setup
#from Cython.Build import cythonize
#
#setup(
#        ext_modules=cythonize("calculateCoefficients.pyx")
#        )

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#
import numpy

#setup(
#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [Extension("multiply",
#                             sources=["multiply.pyx", "c_multiply.c"],
#                             include_dirs=[numpy.get_include()])],
#)

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("calculateCoefficientsTrainExpPointer",
                             sources=["calculateCoefficientsTrainExpPointer.pyx", "c_calculateCoefficientsTrainExpPointer.c"],
                             include_dirs=[numpy.get_include()])],
)
