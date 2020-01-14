# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:28:38 2016

@author: jmascolo

rbf - Radial basis functions for interpolation/smoothing scattered Nd data.

Written by John Travers <jtravs@gmail.com>, February 2007
Based closely on Matlab code by Alex Chirokov
Additional, large, improvements by Robert Hetland
Some additional alterations by Travis Oliphant

Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Copyright (c) 2007, John Travers <jtravs@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of Robert Hetland nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from __future__ import division, print_function, absolute_import

import sys

from numpy import (sqrt, log, asarray, newaxis, all, dot, exp, eye,
                   float_, vstack, hstack, ones, transpose, zeros)
from scipy import linalg
from scipy._lib.six import callable, get_method_function, \
     get_function_code

__all__ = ['Rbf_poly']


class Rbf_poly(object):
    
    def _euclidean_norm(self, x1, x2):
        return sqrt(((x1 - x2)**2).sum(axis=0))

    def _h_multiquadric(self, r):
        return sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_inverse_multiquadric(self, r):
        return 1.0/sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_gaussian(self, r):
        return exp(-(1.0/self.epsilon*r)**2)

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return r**3

    def _h_quintic(self, r):
        return r**5

    def _h_thin_plate(self, r):
        result = r**2 * log(r)
        result[r == 0] = 0  # the spline is zero at zero
        return result
        
    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                functionlist = [x[3:] for x in dir(self) if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                     ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)
        elif callable(self.function):
            allow_one = False
            if hasattr(self.function, 'func_code') or \
                   hasattr(self.function, '__code__'):
                val = self.function
                allow_one = True
            elif hasattr(self.function, "im_func"):
                val = get_method_function(self.function)
            elif hasattr(self.function, "__call__"):
                val = get_method_function(self.function.__call__)
            else:
                raise ValueError("Cannot determine number of arguments to function")

            argcount = get_function_code(val).co_argcount
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                if sys.version_info[0] >= 3:
                    self._function = self.function.__get__(self, Rbf_poly)
                else:
                    import new
                    self._function = new.instancemethod(self.function, self,
                                                        Rbf_poly)
            else:
                raise ValueError("Function argument must take 1 or 2 arguments.")

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of the same shape")
        return a0
        
        
    def __init__(self, *args, **kwargs):
        if len(args)%2 != 0:
            raise ValueError("Output and input points must have the same dimension")
        self.d = int(len(args)/2)
        self.xs = asarray([asarray(a, dtype=float_).flatten() for a in args[:self.d]])
        self.xa = asarray([asarray(a, dtype=float_).flatten() for a in args[self.d:]])
        self.Ns = self.xs.shape[-1]
        self.Na = self.xa.shape[-1]
        
        xs = self.xs[0]
        if self.d > 1:
            for i in range (1, self.d):
                xs = vstack((xs, self.xs[i]))
        self.x_s = xs.transpose()
                
        xa = self.xa[0]
        if self.d > 1:
            for i in range (1, self.d):
                xa = vstack((xa, self.xa[i]))
        self.x_a = xa.transpose()
        
        if not all([x.size == self.xs[0].size for x in self.xs]):
            raise ValueError("All input points arrays must be equal length.")
            
        if not all([x.size == self.xa[0].size for x in self.xa]):
            raise ValueError("All output points arrays must be equal length.")
            
        self.norm = kwargs.pop('norm', self._euclidean_norm)
        rss = self._call_norm(self.xs, self.xs)
        self.epsilon = kwargs.pop('epsilon', None)
        if self.epsilon is None:
            self.epsilon = rss.mean()
            
        self.function = kwargs.pop('function', 'multiquadric')
        
        # attach anything left in kwargs to self
        #  for use by any user-callable function or
        #  to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)
            
        ras = self._call_norm(self.xa, self.xs)
        
        #Matrix definition before assembling
        self.M = self._init_function(rss)
#        print ('det M=',linalg.det(self.M))
        self.M=self.M+eye(self.M.shape[0])*1e-10
        self.Minv = linalg.inv(self.M)
        self.P = vstack((ones((1, self.Ns)), transpose(self.x_s)))
        self.Mp = linalg.inv(self.P.dot(self.Minv).dot(transpose(self.P)))
        Css_up = self.Mp.dot(self.P).dot(self.Minv)
        Css_lo = self.Minv - self.Minv.dot(transpose(self.P)).dot(self.Mp).dot(self.P).dot(self.Minv)
        self.Css_inv = vstack((Css_up, Css_lo))
        
        self.Aas = hstack((ones((self.Na, 1)), self.x_a, self._init_function(ras)))
        
        self.H = self.Aas.dot(self.Css_inv)

        
    def _call_norm(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1[newaxis, :]
        if len(x2.shape) == 1:
            x2 = x2[newaxis, :]
        x1 = x1[..., :, newaxis]
        x2 = x2[..., newaxis, :]
        return self.norm(x1, x2)
        
        
    def __call__(self, *args):
        args = [asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        us = asarray([a.flatten() for a in args], dtype=float_)
        
        u_s = us[0]
        if self.d > 1:
            for i in range (1, self.d):
                u_s = vstack((u_s, us[i]))
        u_s = u_s.transpose()
        
        if self.x_s.shape != u_s.shape:
            raise ValueError("Points and values vectors must have the same shape")
           
        u_a = self.H.dot(u_s)
        return u_a