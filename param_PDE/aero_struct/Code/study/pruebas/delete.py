# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:48:53 2018

@author: ochandre
"""

import numpy as np

nCandidates = 5
indexCand = np.zeros((nCandidates,), dtype=int)
for i in range(nCandidates):
    indexCand[i] = i

candidateMaxError = 0
print indexCand
print candidateMaxError
xC = 0
while True:
    if indexCand[xC] == candidateMaxError:
        break
    xC += 1
print xC
indexCand = np.delete(indexCand,xC)
print indexCand

candidateMaxError = 4
print candidateMaxError
xC = 0
while True:
    if indexCand[xC] == candidateMaxError:
        break
    xC += 1
print xC
indexCand = np.delete(indexCand,xC)
print indexCand

candidateMaxError = 3
print candidateMaxError
xC = 0
while True:
    if indexCand[xC] == candidateMaxError:
        break
    xC += 1
print xC
indexCand = np.delete(indexCand,xC)
print indexCand