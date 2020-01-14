import numpy as np
from scipy import linalg
import functions_Offline as foff
from transfert import transfert_matrix
from mesh_deformation_airbus import mesh_deformation_airbus
from VLM import VLM_study

param_data = np.zeros(9)
param_data[0] = 2.5
param_data[1] = 0.85*np.sqrt(1.4*22552.0/0.3629)
param_data[2] = 0.3629
param_data[3] = 70.*1e9
param_data[4] = 0.3

pCandidate = [[0.00145,0.00145,0.00435,0.00435,34.0,122.0, 5.0],
              [0.030,0.015,0.090,0.090,80.0,845.0, 1.0]]
A = np.zeros((627,627,2))
B = np.zeros((627,2))

n_ribs_1 = 8
n_ribs_2 = 12
foff.create_mesh_files(n_ribs_1,n_ribs_2)

for iCandidate in range(2):
    #### MODIFIER ####
    param_data[0] = pCandidate[iCandidate][6]
    param_data[5] = pCandidate[iCandidate][0]
    param_data[6] = pCandidate[iCandidate][1]
    param_data[7] = pCandidate[iCandidate][2]
    param_data[8] = pCandidate[iCandidate][3]
    b = pCandidate[iCandidate][4]
    S = pCandidate[iCandidate][5]
    ##################
    a_new = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/FEM_mesh.msh') 
    a_new.new_wing_mesh(b,S,phi=(30.*np.pi/180.),diedre=(1.5*np.pi/180.),BF=5.96,Mach=0.85)
    ### b) Defining the files used and the parameters
    vlm_mesh_file = '../mesh/param_wing/new_VLM.msh'
    vlm_mesh_file_out = '../mesh/param_wing/new_VLM_def.msh'
    vlm_part = VLM_study(vlm_mesh_file_out,alpha = param_data[0],v_inf = param_data[1],rho = param_data[2])
    A[:,:,iCandidate] = vlm_part.get_A()
    B[:,iCandidate] = vlm_part.get_B()

resA = A[:,:,1] - A[:,:,0]
resB = B[:,1] - B[:,0]

print("max A ="+str(np.max(resA)))
print("min A ="+str(np.min(resA)))
print("max B ="+str(np.max(resB)))
print("min B ="+str(np.min(resB)))