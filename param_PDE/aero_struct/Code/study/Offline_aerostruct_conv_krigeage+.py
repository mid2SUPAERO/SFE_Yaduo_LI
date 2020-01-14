"""
OFFLINE PHASE aerostruct
This program aims to use the POD-ROM to create a reduced basis able to be used in a real-time application.
It is applied to an aeroelastic problem. A Greedy algorithm has been used to introduced a POD based on 
the snapshots method combined with the Singular Value Decomposition (SVD).
Using version 3.0.6 of GMSH.

July 2018
@author: ochandre
"""

import numpy as np
from scipy import linalg
from pyDOE import lhs
import functions_Offline as foff
import timeit
from sklearn.externals import joblib
import sys
sys.path.append('..')
from gaussian_process import GaussianProcess
from transfert.transfert import transfert_matrix
import shutil
from mesh.mesh_deformation_airbus import mesh_deformation_airbus
from VLM.VLM import VLM_study
import matplotlib.pyplot as plt

##PARAMETERS BOUNDS: h_skins, h_ribs, h_spars_le, h_spars_te, b, S
# h_* = thickness of different parts (m)
# b = wing span (m)
# S = wing surface (m^2)
pMin_i = np.array([0.00145,0.00145,0.00435,0.00435,34.0,122.0])
pMax_i = np.array([0.030,0.015,0.090,0.090,80.0,845.0])
pMin = np.zeros((6,1))
pMin[:,0]= pMin_i[:]
pMax = np.zeros((6,1))
pMax[:,0]= pMax_i[:]
np.save("../results/Offline/pMin",pMin)
np.save("../results/Offline/pMax",pMax)

##OTHERS PARAMETERS OF INTEREST
# alpha = angle of attack
# M_inf = Mach number
# E = Young modulus (Pa)
# nu = Poisson's ratio
# BF = distance of the fuselage over the wing (m)
# phi_d = sweep angle
# diedre_d = dihedral angle 
# n_ribs_1 = number of ribs (first sector)
# n_ribs_2 = number of ribs (second sector)
alpha = 2.5 
M_inf = 0.85
E = 70.*1e9
nu = 0.3
BF = 5.96
phi_d = 30.
diedre_d = 1.5
n_ribs_1 =  8
n_ribs_2 =  12

##FIXED PARAMETERS (Flight Altitude = 11000 meters)
# rho = density at 11000 meters (kg/m^3)
# Pressure = pressure at 11000 meters (Pa)
# adiabatic_i = adiabatic index of air
# a_speed = speed of sound at 11000 meters (m/s)
# phi = sweep angle (rad)
# diedre = dihedral angle (rad)
rho = 0.3629
Pressure = 22552.
adiabiatic_i = 1.4
a_speed = np.sqrt(adiabiatic_i*Pressure/rho)
v_inf = M_inf*a_speed
phi = phi_d*np.pi/180.
diedre = diedre_d*np.pi/180.

##CREATION OF MESH FILES VLM AND FEM
foff.create_mesh_files(n_ribs_1,n_ribs_2)

param_data = np.zeros(9)
param_data[0] = alpha
param_data[1] = v_inf
param_data[2] = rho
param_data[3] = E
param_data[4] = nu
np.save("../results/Offline/param_data",param_data)

##GREEDY ALGORITHM
tic = timeit.default_timer()
# 0) Initialization
nSamples = 50
nIterations = nSamples-1
nCandidates = 150
nParam = len(pMin)
pCandidate = np.dot(pMin,np.ones((1,nCandidates))) + lhs(nParam,nCandidates).T*np.dot((pMax-pMin),np.ones((1,nCandidates)))
indexCand = np.zeros((nCandidates,), dtype=int)
for i in range(nCandidates):
    indexCand[i] = i
pCandMax = np.zeros((nParam,nSamples))
np.save("../results/Offline/pCandidate",pCandidate)
np.save("../results/Offline/nCandidates",nCandidates)
np.save("../results/Offline/nSamples",nSamples)
# 1) Ramdomly select a first sample (in the middle of the domain)
pMiddle = (pMin+pMax)/2
param_data[5] = pMiddle[0,0]
param_data[6] = pMiddle[1,0]
param_data[7] = pMiddle[2,0]
param_data[8] = pMiddle[3,0]
b = pMiddle[4,0]
S = pMiddle[5,0]
pCandMax[:,0] = pMiddle[:,0]
# Calculating the number of nodes and initialization of samples matrices
n_VLM_nodes, n_FEM_nodes, n_gamma_nodes = foff.calcule_nodes(param_data)
uSamples = np.zeros((n_FEM_nodes*6,nSamples)) # POD displacement
gSamples = np.zeros((n_gamma_nodes,nSamples)) # POD circulation
# 2) Solving HDM-based problem
print "Greedy Iteration num.: 0\n",
uMiddle,gMiddle = foff.run_aerostruct(param_data,b,S,phi,diedre,BF,Mach=M_inf)
uSamples[:,0] = uMiddle
gSamples[:,0] = gMiddle
# 3) Building the ROB
uV,_,_ = linalg.svd(uSamples,full_matrices=False)
gV,_,_ = linalg.svd(gSamples,full_matrices=False)
uV_tr = np.transpose(uV)
gV_tr = np.transpose(gV)
# 4) Iteration loop

err_ok = 0
err_ko = 0
err_max = 0

for iIteration in range(nIterations):
    print "Greedy Iteration num.: ",iIteration+1
    # 5) Solve argmax(residual)
    maxError = 0.
    candidateMaxError = 0
    for iCandidate in indexCand:
        # 5) Loading candidates information
        param_data[5] = pCandidate[0,iCandidate]
        param_data[6] = pCandidate[1,iCandidate]
        param_data[7] = pCandidate[2,iCandidate]
        param_data[8] = pCandidate[3,iCandidate]
        b = pCandidate[4,iCandidate]
        S = pCandidate[5,iCandidate]
        ###Fixed point convergence
        err = 1.0
        err_vec = []
        tol = 5.0e-2
        itermax = 150
        n = 0
        q_0 = np.ones((nSamples,))*1.0
        g_0 = np.ones((nSamples,))*1.0
        tic1 = timeit.default_timer()
        while n < itermax and err > tol:
            ## Definition of the mesh...
            ### a) Applying the right values in the wing
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
            print('Error = '+str(err))
            err_vec = np.append(err_vec,err)
            q_0 = q.copy()
            g_0 = gr.copy()
            n += 1
        if n == itermax:
            err_ko += 1
            if err > err_max: err_max = err
        else:
            err_ok += 1
        # 5b) Compute error indicator
        toc1 = timeit.default_timer()
        print("Reduced convergence time = "+str(toc1-tic1)+"s")
        print("n = "+str(n))
#        plt.plot(np.linspace(1,n,n),err_vec)
#        plt.ylim(top=1e-1)
#        plt.ylim(bottom=1e-4)
#        plt.grid(True)
#        plt.show()
        
        Y1 = np.dot(K,np.dot(uV,q))- Fs
        errorIndicator = np.dot(np.transpose(Y1),Y1)
        if (iCandidate == 0 or maxError < errorIndicator):
            candidateMaxError = iCandidate
            maxError = errorIndicator
    # 6) Solving HDM-based problem for the candidate with the highest error indicator
    param_data[5] = pCandidate[0,candidateMaxError]
    param_data[6] = pCandidate[1,candidateMaxError]
    param_data[7] = pCandidate[2,candidateMaxError]
    param_data[8] = pCandidate[3,candidateMaxError]
    b = pCandidate[4,candidateMaxError]
    S = pCandidate[5,candidateMaxError]
    pCandMax[:,iIteration+1] = pCandidate[:,candidateMaxError]
    uIter,gIter = foff.run_aerostruct(param_data,b,S,phi,diedre,BF,Mach=M_inf)
    uSamples[:,iIteration+1] = uIter
    gSamples[:,iIteration+1] = gIter
    # 7) Building the ROB
    uV,_,_ = linalg.svd(uSamples,full_matrices=False)
    gV,_,_ = linalg.svd(gSamples,full_matrices=False)
    uV_tr = np.transpose(uV)
    gV_tr = np.transpose(gV)
    ##UPDATING indexCand: The current index must be removed!
    xC = 0
    while True:
        if indexCand[xC] == candidateMaxError:
            break
        xC += 1
    indexCand = np.delete(indexCand,xC)

##SAVE OFFLINE DATA
u_name_V = "../results/Offline/uV"
g_name_V = "../results/Offline/gV"
np.save(u_name_V,uV)
np.save(g_name_V,gV)
toc = timeit.default_timer()
print("TOTAL COMPUTATION TIME: "+str((toc-tic)/60.)+" min")
print("total ok = "+str(err_ok))
print("total ko = "+str(err_ko))

##KRIGING
# Initialization
tic = timeit.default_timer()
U_comp,G_comp,pCandMax = foff.kriging_extended_B(uSamples,gSamples,pCandMax,pMin,pMax, nSamples, param_data, uV, gV, phi, diedre, BF, M_inf)
np.save("../results/Offline/U_comp",U_comp)
np.save("../results/Offline/G_comp",G_comp)
# Weight calculations
nLHS = len(U_comp[0,:])
a_kri = np.zeros((nLHS,nSamples))
b_kri = np.zeros((nLHS,nSamples))
for i in range(nSamples):
    for j in range(nLHS):
        a_kri[j,i] = np.dot(U_comp[:,j].T,uV[:,i])
        b_kri[j,i] = np.dot(G_comp[:,j].T,gV[:,i])
np.save("../results/Offline/a_kri",a_kri)
np.save("../results/Offline/b_kri",b_kri)
# Trained solution
mean = "constant"
covariance = "squared_exponential"
theta_U = np.array([100000.0]*6)
theta_L = np.array([0.001]*6)
theta_0 = np.array([1.0]*6)
for i in range(nSamples):
    GP_u = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
    GP_u.fit(pCandMax.T, a_kri[:,i])
    GP_g = GaussianProcess(regr = mean, corr = covariance,theta0 = theta_0,thetaL = theta_L, thetaU = theta_U)
    GP_g.fit(pCandMax.T, b_kri[:,i])
    joblib.dump(GP_u, "../results/Offline/GP_alpha_"+str(i)+".pkl")
    joblib.dump(GP_g, "../results/Offline/GP_beta_"+str(i)+".pkl")
    
toc = timeit.default_timer()
print("KRIGING COMPUTATION TIME: "+str((toc-tic)/60.)+" min")