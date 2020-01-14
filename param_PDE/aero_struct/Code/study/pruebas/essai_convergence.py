# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:17:18 2019

@author: o.chandre-vila
"""

import numpy as np
from scipy import linalg
import functions_Offline as foff
import timeit
from transfert import transfert_matrix
import shutil
from mesh_deformation_airbus import mesh_deformation_airbus
from VLM import VLM_study

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
param_data = np.zeros(9)
param_data[0] = alpha
param_data[1] = v_inf
param_data[2] = rho
param_data[3] = E
param_data[4] = nu
param_data[5] = 0.00145
param_data[6] = 0.00145
param_data[7] = 0.00435
param_data[8] = 0.00435
nSamples = 10
nCandidates = 50
nParam = 6
b = 34.0
S = 122.0


foff.create_mesh_files(n_ribs_1,n_ribs_2)

n_VLM_nodes, n_FEM_nodes, n_gamma_nodes = foff.calcule_nodes(param_data)
uSamples = np.zeros((n_FEM_nodes*6,nSamples)) # POD displacement
gSamples = np.zeros((n_gamma_nodes,nSamples)) # POD circulation
uV,_,_ = linalg.svd(uSamples,full_matrices=False)
gV,_,_ = linalg.svd(gSamples,full_matrices=False)

err = 1.0
tol = 1e-3
itermax = 100
n = 0
q_0 = np.ones((nSamples,))*1000.0
g_0 = np.ones((nSamples,))*1000.0
uV_tr = np.transpose(uV)
gV_tr = np.transpose(gV)


while n < itermax and err > tol:
    # 5) Loading candidates information
    # Definition of the mesh...
    ## a) Applying the right values in the wing
    a_new = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/FEM_mesh.msh') 
    a_new.new_wing_mesh(b,S,phi,diedre,BF,Mach=M_inf)
    ### b) Defining the files used and the parameters
    vlm_mesh_file = '../mesh/param_wing/new_VLM.msh'
    if n == 0:
        vlm_mesh_file_out = '../mesh/param_wing/new_VLM_def.msh'
        shutil.copyfile(vlm_mesh_file,vlm_mesh_file_out)
    else:
        my_vlm = VLM_study(vlm_mesh_file_out,alpha = param_data[0],v_inf = param_data[1],rho = param_data[2])
        my_vlm.deformed_mesh_file(new_nodes,vlm_mesh_file_out)
    ### c) Performing the analysis and saving the results
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
    ## Loop corroboration
    H = transfert_matrix(fem_nodes,vlm_nodes,function_type='thin_plate')
    q_a = np.zeros((len(vlm_nodes)*6,))
    for i in range(6):
        q_a[0+i::6] = np.dot(H,q_exp[6*(fem_nodes_ind-1)+i])
    ### VLM mesh deformation 
    new_nodes = vlm_nodes_tot.copy()
    new_nodes[:,1] = new_nodes[:,1] + q_a[0::6]
    new_nodes[:,2] = new_nodes[:,2] + q_a[1::6]
    new_nodes[:,3] = new_nodes[:,3] + q_a[2::6]
    ### Compute relative error
    err = np.max([np.linalg.norm(q_0-q)/np.linalg.norm(q),np.linalg.norm(g_0-gr)/np.linalg.norm(gr)])
    q_0 = q.copy()
    g_0 = gr.copy()
    n += 1