# -*- coding: utf-8 -*-
"""
Converging time analysis.

Created on Mon Feb 04 17:58:47 2019

@author: o.chandre-vila
"""
import numpy as np
from scipy import linalg
from pyDOE import lhs
import functions_Offline as foff
import timeit
from transfert import transfert_matrix
import shutil
from mesh_deformation_airbus import mesh_deformation_airbus
from VLM import VLM_study

# DATA INPUTS
alpha = 2.5 
M_inf = 0.85
E = 70.*1e9
nu = 0.3
BF = 5.96
phi_d = 30.
diedre_d = 1.5
n_ribs_1 =  8
n_ribs_2 =  12
rho = 0.3629
Pressure = 22552.
adiabiatic_i = 1.4
a_speed = np.sqrt(adiabiatic_i*Pressure/rho)
v_inf = M_inf*a_speed
phi = phi_d*np.pi/180.
diedre = diedre_d*np.pi/180.
foff.create_mesh_files(n_ribs_1,n_ribs_2)
param_data = np.zeros(9)
param_data[0] = alpha
param_data[1] = v_inf
param_data[2] = rho
param_data[3] = E
param_data[4] = nu
param_data[5] = 0.015725
param_data[6] = 0.008225
param_data[7] = 0.047175
param_data[8] = 0.0477175
b = 57.
S = 483.5
u,g = foff.run_aerostruct(param_data,b,S,phi,diedre,BF,Mach=M_inf)
n_VLM_nodes, n_FEM_nodes, n_gamma_nodes = foff.calcule_nodes(param_data)
uS = np.zeros((n_FEM_nodes*6,3))
gS = np.zeros((n_gamma_nodes,3))
uS[:,0] = u
gS[:,0] = g
uV,_,_ = linalg.svd(uS,full_matrices=False)
gV,_,_ = linalg.svd(gS,full_matrices=False)
uV_tr = np.transpose(uV)
gV_tr = np.transpose(gV)

# METHODS EVALUATION
pMin_i = np.array([0.00145,0.00145,0.00435,0.00435,34.0,122.0])
pMax_i = np.array([0.030,0.015,0.090,0.090,80.0,845.0])
pMin = np.zeros((6,1))
pMin[:,0]= pMin_i[:]
pMax = np.zeros((6,1))
pMax[:,0]= pMax_i[:]
nParam = len(pMin)
nCand = 25
pCand = np.dot(pMin,np.ones((1,nCand))) + lhs(nParam,nCand).T*np.dot((pMax-pMin),np.ones((1,nCand)))
iCand = np.zeros((nCand,), dtype=int)
for i in range(nCand): iCand[i] = i
## 1) Original problem
print('-------------------ORIGINAL PROBLEM----------------------')
t_orig = []
for iC in iCand:
    param_data[5] = pCand[0,iC]
    param_data[6] = pCand[1,iC]
    param_data[7] = pCand[2,iC]
    param_data[8] = pCand[3,iC]
    b = pCand[4,iC]
    S = pCand[5,iC]
    tic1 = timeit.default_timer()
    u,g = foff.run_aerostruct(param_data,b,S,phi,diedre,BF,Mach=M_inf)
    toc1 = timeit.default_timer()
    t_orig = np.hstack((t_orig,(toc1-tic1)))
    
print('Min = '+str(np.amin(t_orig))+' s')
print('Max = '+str(np.amax(t_orig))+' s')
print('Mean = '+str(np.mean(t_orig))+' s')

## 2) Non loop problem
print('-------------------NON LOOP PROBLEM----------------------')
t_n_loop = []
for iC in iCand:
    param_data[5] = pCand[0,iC]
    param_data[6] = pCand[1,iC]
    param_data[7] = pCand[2,iC]
    param_data[8] = pCand[3,iC]
    b = pCand[4,iC]
    S = pCand[5,iC]
    tic2 = timeit.default_timer()
    A, B = foff.VLM_params(param_data,b,S,phi,diedre,BF,Mach=M_inf)
    Ar = np.dot(np.dot(gV_tr,A),gV)
    Br = np.dot(gV_tr,B)
    Ar_lu = linalg.lu_factor(Ar)
    gr = linalg.lu_solve(Ar_lu,-Br)
    g_exp = np.dot(gV,gr)
    F_s,_,_ = foff.get_Fs(g_exp,param_data)
    K, Fs,_ = foff.get_FEM(param_data,F_s)
    Kr = np.dot(np.dot(uV_tr,K),uV)
    Fr = np.dot(uV_tr,Fs)
    q = linalg.solve(Kr,Fr)
    toc2 = timeit.default_timer()
    t_n_loop = np.hstack((t_n_loop,(toc2-tic2)))

print('Min = '+str(np.amin(t_n_loop))+' s')
print('Max = '+str(np.amax(t_n_loop))+' s')
print('Mean = '+str(np.mean(t_n_loop))+' s')

## 3) Loop problem
print('--------------------LOOP PROBLEM----------------------')
t_loop = []
for iC in iCand:
    param_data[5] = pCand[0,iC]
    param_data[6] = pCand[1,iC]
    param_data[7] = pCand[2,iC]
    param_data[8] = pCand[3,iC]
    b = pCand[4,iC]
    S = pCand[5,iC]
    tic3 = timeit.default_timer()
    err = 1.0
    tol = 1.5e-2
    itermax = 10
    n = 0
    q_0 = np.ones((3,))*1000.0
    g_0 = np.ones((3,))*1000.0
    while n < itermax and err > tol:
        a_new = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/FEM_mesh.msh') 
        a_new.new_wing_mesh(b,S,phi,diedre,BF,Mach=M_inf)
        vlm_mesh_file = '../mesh/param_wing/new_VLM.msh'
        if n == 0:
            vlm_mesh_file_out = '../mesh/param_wing/new_VLM_def.msh'
            shutil.copyfile(vlm_mesh_file,vlm_mesh_file_out)
        else:
            my_vlm = VLM_study(vlm_mesh_file_out,alpha = param_data[0],v_inf = param_data[1],rho = param_data[2])
            my_vlm.deformed_mesh_file(new_nodes,vlm_mesh_file_out)
        vlm_part = VLM_study(vlm_mesh_file_out,alpha = param_data[0],v_inf = param_data[1],rho = param_data[2])
        A = vlm_part.get_A()
        B = vlm_part.get_B()
        vlm_part.read_gmsh_mesh()
        vlm_nodes_tot = vlm_part.nodes.copy()
        Ar = np.dot(np.dot(gV_tr,A),gV)
        Br = np.dot(gV_tr,B)
        Ar_lu = linalg.lu_factor(Ar)
        gr = linalg.lu_solve(Ar_lu,-Br)
        g_exp = np.dot(gV,gr)
        F_s, fem_nodes, vlm_nodes = foff.get_Fs(g_exp,param_data)
        K, Fs, fem_nodes_ind = foff.get_FEM(param_data,F_s)
        Kr = np.dot(np.dot(uV_tr,K),uV)
        Fr = np.dot(uV_tr,Fs)
        q = linalg.solve(Kr,Fr)
        q_exp = np.dot(uV,q)
        H = transfert_matrix(fem_nodes,vlm_nodes,function_type='thin_plate')
        q_a = np.zeros((len(vlm_nodes)*6,))
        for i in range(6):
            q_a[0+i::6] = np.dot(H,q_exp[6*(fem_nodes_ind-1)+i])
        new_nodes = vlm_nodes_tot.copy()
        new_nodes[:,1] = new_nodes[:,1] + q_a[0::6]
        new_nodes[:,2] = new_nodes[:,2] + q_a[1::6]
        new_nodes[:,3] = new_nodes[:,3] + q_a[2::6]
        err = np.max([np.linalg.norm(q_0-q)/np.linalg.norm(q),np.linalg.norm(g_0-gr)/np.linalg.norm(gr)])
        q_0 = q.copy()
        g_0 = gr.copy()
        n += 1
    toc3 = timeit.default_timer()
    t_loop =np.hstack((t_loop,(toc3-tic3)))
    
print('Min = '+str(np.amin(t_loop))+' s')
print('Max = '+str(np.amax(t_loop))+' s')
print('Mean = '+str(np.mean(t_loop))+' s')