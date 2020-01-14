# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:23:13 2018

@author: ochandre
"""
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

pMin_i = np.array([0.028,0.009,0.017,0.009,64.75,443.56])
pMax_i = np.array([0.031,0.015,0.025,0.015,75.65,500.21])
pMin = np.zeros((6,1))
pMin[:,0]= pMin_i[:]
pMax = np.zeros((6,1))
pMax[:,0]= pMax_i[:]
nCandidates = 50
nParam = len(pMin)
pCandidate = np.dot(pMin,np.ones((1,nCandidates))) + lhs(nParam,nCandidates).T*np.dot((pMax-pMin),np.ones((1,nCandidates)))
i_fig = 0
for i in range(6):
    i_fig += 1
    plt.figure(i_fig)
    plt.plot(pCandidate[i,:])
    plt.grid(True)
    plt.plot(np.linspace(0, nCandidates-1, nCandidates),pMin[i]*np.ones((nCandidates,1)),'r')
    plt.plot(np.linspace(0, nCandidates-1, nCandidates),pMax[i]*np.ones((nCandidates,1)),'r')
    
pCandidate = np.dot(pMin,np.ones((1,nCandidates))) + np.random.uniform(low=0.0, high=1.0, size=(nParam,nCandidates))*np.dot((pMax-pMin),np.ones((1,nCandidates)))
for i in range(6):
    i_fig += 1
    plt.figure(i_fig)
    plt.plot(pCandidate[i,:],'g')
    plt.grid(True)
    plt.plot(np.linspace(0, nCandidates-1, nCandidates),pMin[i]*np.ones((nCandidates,1)),'r')
    plt.plot(np.linspace(0, nCandidates-1, nCandidates),pMax[i]*np.ones((nCandidates,1)),'r')