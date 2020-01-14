# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:11:06 2018

@author: ochandre
"""

import numpy as np
from sklearn.externals import joblib
import functions_Offline as foff
from mesh_deformation_airbus import mesh_deformation_airbus
from VLM import VLM_study
from FEM import FEM_study
import timeit

##LOADING... (FROM OFFLINE PROCESS)
# 1) Problem data
pMin = np.load("../results/Offline/pMin.npy")
pMax = np.load("../results/Offline/pMax.npy")
nCandidates = np.load("../results/Offline/nCandidates.npy")
nSamples = np.load("../results/Offline/nSamples.npy")
param_data = np.load("../results/Offline/param_data.npy")
pCandidate = np.load("../results/Offline/pCandidate.npy")
# 2) RB data
iIteration = np.load("../results/Offline/last_iIteration.npy")
u_name = "../results/Offline/uV"+str(iIteration+1)+".npy"
g_name = "../results/Offline/gV"+str(iIteration+1)+".npy"
uV = np.load(u_name)
gV = np.load(g_name)

##ONLINE TREATMENT
tic = timeit.default_timer()
#---------CONF 1------------
h_skins = 0.028
h_ribs = 0.009
h_spars_le = 0.017
h_spars_te = 0.009
b = 64.75
S = 443.56
#---------CONF 2------------
#h_skins = 0.029
#h_ribs = 0.014
#h_spars_le = 0.019
#h_spars_te = 0.013
#b = 66.83
#S = 449.7
#---------CONF 3------------
#h_skins = 0.030
#h_ribs = 0.01
#h_spars_le = 0.022
#h_spars_te = 0.011
#b = 73.20
#S = 444.44
#---------CONF 4------------
#h_skins = 0.031
#h_ribs = 0.015
#h_spars_le = 0.025
#h_spars_te = 0.015
#b = 75.65
#S = 500.21
X = np.array([h_skins,h_ribs,h_spars_le,h_spars_te,b,S])
pC = np.zeros((6))
pC[0] = param_data[5] = h_skins
pC[1] = param_data[6] = h_ribs
pC[2] = param_data[7] = h_spars_le
pC[3] = param_data[8] = h_spars_te
pC[4] = b
pC[5] = S
# 1) Krigeage prediction of the interested point
u_mp = u_vp = gamma_mp = gamma_vp = np.zeros((1,nSamples))
for i in range(nSamples):
    # *Loading Krigeage data
    name_a = joblib.load("../results/Offline/GP_alpha_"+str(i)+".pkl")
    name_b = joblib.load("../results/Offline/GP_beta_"+str(i)+".pkl")
    # **Prediction
    u_mp[0,i], u_vp[0,i] = name_a.predict(X.reshape((1,6)), eval_MSE=True)
    gamma_mp[0,i], gamma_vp[0,i] = name_b.predict(X.reshape((1,6)), eval_MSE=True)

u_mean = 0
u_var2 = 0
gamma_mean = 0
gamma_var2 = 0
for i in range(nSamples):
    u_mean += np.dot(u_mp[0,i],uV[:,i])
    u_var2 += np.dot(np.square(u_vp[0,i]),np.square(uV[:,i]))
    gamma_mean += np.dot(gamma_mp[0,i],gV[:,i])
    gamma_var2 += np.dot(np.square(gamma_vp[0,i]),np.square(gV[:,i]))
# 2) Calculation of u_est and g_est
u_est = u_mean
u_est1 = u_mean+3.0*np.sqrt(u_var2)
u_est2 = u_mean-3.0*np.sqrt(u_var2)
g_est = gamma_mean
g_est1 = gamma_mean+3.0*np.sqrt(gamma_var2)
g_est2 = gamma_mean-3.0*np.sqrt(gamma_var2)
# 3) Publication of the results: u_est, g_est
a_new = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/FEM_mesh.msh') 
a_new.new_wing_mesh(b,S)
vlm_mesh = '../mesh/param_wing/new_VLM.msh'
fem_mesh = '../mesh/param_wing/new_FEM.msh'
n_VLM_nodes, n_FEM_nodes, n_gamma_nodes = foff.calcule_nodes(pC)
# 3a) Computation of strains and stress
element_property,material,element_type = foff.create_dict(param_data)
my_fem = FEM_study(fem_mesh,element_type,element_property,material)
my_fem.read_mesh_file()
strain_dict, stress_dict = my_fem.get_strain_and_stress(u_est)
# 3b) u_est
my_fem.post_processing(u_est,"../results/Param_wing/u_est")
# 3c) strain and stress
my_fem.post_processing_strain_stress(strain_dict,stress_dict,"../results/Param_wing/est")
# 3d) g_est
my_vlm = VLM_study(vlm_mesh)
my_vlm.post_processing_gamma('Param_wing/g_est',g_est)