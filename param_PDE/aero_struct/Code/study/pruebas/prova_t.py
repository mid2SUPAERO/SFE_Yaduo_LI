import numpy as np
from scipy import linalg
import functions_Offline as foff
import timeit
from transfert import transfert_matrix
import shutil
from mesh_deformation_airbus import mesh_deformation_airbus
from VLM import VLM_study

param_data = np.zeros(9)
param_data[0] = 2.5
param_data[1] = 0.85*np.sqrt(1.4*22552.0/0.3629)
param_data[2] = 0.3629
param_data[3] = 70.*1e9
param_data[4] = 0.3
param_data[5] = 0.005
param_data[6] = 0.007
param_data[7] = 0.008
param_data[8] = 0.004
b = 34.0
S = 122.0

nSamples = 10
nIterations = nSamples-1
nCandidates = 50
n_FEM_nodes = 498
n_gamma_nodes = 627

###Fixed point convergence
err = 1.0
err_vec = []
tol = 1.0e-2
itermax = 100
n = 0
q_0 = np.ones((nSamples,))*1.0
g_0 = np.ones((nSamples,))*1.0

uSamples = np.random.rand(n_FEM_nodes*6,nSamples)
gSamples = np.random.rand(n_gamma_nodes,nSamples)
uV,_,_ = linalg.svd(uSamples,full_matrices=False)
gV,_,_ = linalg.svd(gSamples,full_matrices=False)
uV_tr = np.transpose(uV)
gV_tr = np.transpose(gV)

while n < itermax and err > tol:
    ## Definition of the mesh...
    ### a) Applying the right values in the wing
    tic1 = timeit.default_timer()
    a_new = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/FEM_mesh.msh') 
    a_new.new_wing_mesh(b,S,phi=(30.*np.pi/180.),diedre=(1.5*np.pi/180.),BF=5.96,Mach=0.85)
    ### b) Defining the files used and the parameters
    vlm_mesh_file = '../mesh/param_wing/new_VLM.msh'
    if n == 0:
        vlm_mesh_file_out = '../mesh/param_wing/new_VLM_def.msh'
        shutil.copyfile(vlm_mesh_file,vlm_mesh_file_out)
    else:
        my_vlm = VLM_study(vlm_mesh_file_out,alpha = param_data[0],v_inf = param_data[1],rho = param_data[2])
        my_vlm.deformed_mesh_file(new_nodes,vlm_mesh_file_out)
    ### c) Performing the analysis and saving the results
    toc1 = timeit.default_timer()
    print('temps creacio mesh = '+str(toc1-tic1))
    tic2 = timeit.default_timer()
    vlm_part = VLM_study(vlm_mesh_file_out,alpha = param_data[0],v_inf = param_data[1],rho = param_data[2])
    A = vlm_part.get_A()
    B = vlm_part.get_B()
    vlm_part.read_gmsh_mesh()
    vlm_nodes_tot = vlm_part.nodes.copy()
    toc2 = timeit.default_timer()
    print('temps A i B = '+str(toc2-tic2))
    tic3 = timeit.default_timer()
    Ar = np.dot(np.dot(gV_tr,A),gV)
    Br = np.dot(gV_tr,B)
    Ar_lu = linalg.lu_factor(Ar)
    gr = linalg.lu_solve(Ar_lu,-Br)
    g_exp = np.dot(gV,gr)
    F_s, fem_nodes, vlm_nodes = foff.get_Fs(g_exp,param_data)
    tic5 = timeit.default_timer()
    K, Fs, fem_nodes_ind = foff.get_FEM(param_data,F_s)
    toc5 = timeit.default_timer()
    print('temps K = '+str(toc5-tic5))
    Kr = np.dot(np.dot(uV_tr,K),uV)
    Fr = np.dot(uV_tr,Fs)
    q = linalg.solve(Kr,Fr)
    q_exp = np.dot(uV,q)
    toc3 = timeit.default_timer()
    print('temps problema reduit = '+str(toc3-tic3))
    tic4 = timeit.default_timer()
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
    toc4 = timeit.default_timer()
    print('temps calcul H i recalcul mesh = '+str(toc4-tic4))
    ### Compute relative error
    err = np.max([np.linalg.norm(q_0-q)/np.linalg.norm(q),np.linalg.norm(g_0-gr)/np.linalg.norm(gr)])
    err_vec = np.append(err_vec,err)
    q_0 = q.copy()
    g_0 = gr.copy()
    n += 1
