# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:51:47 2018

@author: ochandre
"""
import numpy as np
#import foff
#import scipy.linalg as linalg
import timeit

K = np.load("K.npy")
F = np.load("Fs.npy")

print "TEST HERMITIAN"
print np.all(K == np.conjugate(np.transpose(K)))
print "TEST SYMMETRY"
print np.allclose(K, K.T, atol=1e-8)
print "TEST POSITIVE-DEFINITE"
print np.all(np.linalg.eigvals(K) > 0)

tic = timeit.default_timer()
#Do stuff
toc = timeit.default_timer()
print toc-tic
