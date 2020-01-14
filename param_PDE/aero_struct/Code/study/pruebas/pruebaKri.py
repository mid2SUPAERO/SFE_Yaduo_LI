# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 17:17:36 2018

@author: ochandre
"""
import numpy as np
import functions_Offline as foff
import timeit
from moe.onera_gp import GaussianProcess

from mesh_deformation_airbus import mesh_deformation_airbus
from VLM import VLM_study
from FEM import FEM_study

nSamples = np.load("../results/Offline/nSamples.npy")
param_data = np.load("../results/Offline/param_data.npy")
iIteration = np.load("../results/Offline/last_iIteration.npy")
u_name = "../results/Offline/uV"+str(iIteration+1)+".npy"
g_name = "../results/Offline/gV"+str(iIteration+1)+".npy"
uV = np.load(u_name)
gV = np.load(g_name)
pMin = np.load("../results/Offline/pMin.npy")
pMax = np.load("../results/Offline/pMax.npy")
param_data = np.load("../results/Offline/param_data.npy")
U_comp = np.load("../results/Offline/U_comp.npy")
G_comp = np.load("../results/Offline/G_comp.npy")
pCandidate = np.load("../results/Offline/pCandidate.npy")
nCandidates = np.load("../results/Offline/nCandidates.npy")
nSamples = np.load("../results/Offline/nSamples.npy")

##KRIGEAGE
tic = timeit.default_timer()
# Weight calculations
a_kri = np.zeros((nCandidates,nSamples))
b_kri = np.zeros((nCandidates,nSamples))
for i in range(nSamples):
    for j in range(nCandidates):
        a_kri[j,i] = np.dot(U_comp[:,j].T,uV[:,i])
        b_kri[j,i] = np.dot(G_comp[:,j].T,gV[:,i])

# Trained solution
mean = "constant"
covariance = "squared_exponential"
theta_U = np.array([100000.0]*6)
theta_L = np.array([0.001]*6)
theta_0 = np.array([1.0]*6)

u0 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
u0.fit(pCandidate.T, a_kri[:,0])

u1 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
u1.fit(pCandidate.T, a_kri[:,1])

u2 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
u2.fit(pCandidate.T, a_kri[:,2])

u3 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
u3.fit(pCandidate.T, a_kri[:,3])

u4 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
u4.fit(pCandidate.T, a_kri[:,4])

g0 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
g0.fit(pCandidate.T, b_kri[:,0])

g1 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
g1.fit(pCandidate.T, b_kri[:,1])

g2 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
g2.fit(pCandidate.T, b_kri[:,2])

g3 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
g3.fit(pCandidate.T, b_kri[:,3])

g4 = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
g4.fit(pCandidate.T, b_kri[:,4])

toc = timeit.default_timer()
print("Krigeage: "+str(toc-tic)+"s")

##ONLINE TREATMENT
tic = timeit.default_timer()
#---------CONF 1------------
#h_skins = 0.028
#h_ribs = 0.009
#h_spars_le = 0.017
#h_spars_te = 0.009
#b = 64.75
#S = 443.56
#---------CONF 2------------
#h_skins = 0.029
#h_ribs = 0.014
#h_spars_le = 0.019
#h_spars_te = 0.013
#b = 66.83
#S = 449.7
#---------CONF 3------------
h_skins = 0.030
h_ribs = 0.01
h_spars_le = 0.022
h_spars_te = 0.011
b = 73.20
S = 444.44
#---------CONF 4------------
#h_skins = 0.031
#h_ribs = 0.015
#h_spars_le = 0.025
#h_spars_te = 0.015
#b = 75.65
#S = 500.21
x_conf = np.array([h_skins,h_ribs,h_spars_le,h_spars_te,b,S])
pC = np.zeros((6))
pC[0] = param_data[5] = h_skins
pC[1] = param_data[6] = h_ribs
pC[2] = param_data[7] = h_spars_le
pC[3] = param_data[8] = h_spars_te
pC[4] = b
pC[5] = S
# 1) Krigeage prediction of the interested point
u_mp = np.zeros((1,nSamples))
u_vp = np.zeros((1,nSamples))
gamma_mp = np.zeros((1,nSamples))
gamma_vp = np.zeros((1,nSamples))

u_mp[0,0],u_vp[0,0] = u0.predict(x_conf.reshape((1,6)), eval_MSE=True)
u_mp[0,1],u_vp[0,1] = u1.predict(x_conf.reshape((1,6)), eval_MSE=True)
u_mp[0,2],u_vp[0,2] = u2.predict(x_conf.reshape((1,6)), eval_MSE=True)
u_mp[0,3],u_vp[0,3] = u3.predict(x_conf.reshape((1,6)), eval_MSE=True)
u_mp[0,4],u_vp[0,4] = u4.predict(x_conf.reshape((1,6)), eval_MSE=True)
gamma_mp[0,0],gamma_vp[0,0] = g0.predict(x_conf.reshape((1,6)), eval_MSE=True)
gamma_mp[0,1],gamma_vp[0,1] = g1.predict(x_conf.reshape((1,6)), eval_MSE=True)
gamma_mp[0,2],gamma_vp[0,2] = g2.predict(x_conf.reshape((1,6)), eval_MSE=True)
gamma_mp[0,3],gamma_vp[0,3] = g3.predict(x_conf.reshape((1,6)), eval_MSE=True)
gamma_mp[0,4],gamma_vp[0,4] = g4.predict(x_conf.reshape((1,6)), eval_MSE=True)
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
#u_est1 = u_mean+3.0*np.sqrt(u_var2)
#u_est2 = u_mean-3.0*np.sqrt(u_var2)
g_est = gamma_mean
#g_est1 = gamma_mean+3.0*np.sqrt(gamma_var2)
#g_est2 = gamma_mean-3.0*np.sqrt(gamma_var2)
toc = timeit.default_timer()
print("ONLINE COMPUTATION TIME: "+str(toc-tic)+" s")
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
#strain_dict, stress_dict = my_fem.get_strain_and_stress(u_est)
# 3b) u_est
my_fem.post_processing(u_est,"../results/Param_wing/u_est")
## 3c) strain and stress
#my_fem.post_processing_strain_stress(strain_dict,stress_dict,"../results/Param_wing/est")
# 3d) g_est
my_vlm = VLM_study(vlm_mesh)
my_vlm.post_processing_gamma('Param_wing/g_est',g_est)