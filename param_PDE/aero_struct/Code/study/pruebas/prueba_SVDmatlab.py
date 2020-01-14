# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:19:17 2018

@author: ochandre
"""
import numpy as np
from scipy import linalg

A = np.array([[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18,19,20,21,22,23],[24,25,26,27,28,29],[30,31,32,33,34,35]])
B = np.array([[0,1,2,3,4,5],[6,7,8,9,10,11]])
C = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]])

Us,Ss,Vhs = linalg.svd(C,full_matrices=False)
#Ss = linalg.diagsvd(Ss,6,2)
Ss = np.diag(Ss)
print np.shape(Us)
print np.shape(Ss)
print np.shape(Vhs)