"""
OFFLINE PHASE aerostruct
This program aims to use the POD-ROM to create a reduced basis able to be used in a real-time application.
It is applied to an aeroelastic problem.

July 2018
@author: ochandre
"""

import numpy as np
from define_geo import airbus_wing
from create_FEM_mesh import *
from create_VLM_mesh import *
import subprocess
from PARAM_WING_VLM_FEM import PARAM_WING_VLM_FEM
from mesh_deformation_airbus import *


##PARAMETERS OF INTEREST
# alpha = angle of attack (°)
# M_inf = Mach number
# E = Young modulus (Pa)
# nu = Poisson's ratio
# h_* = thickness of different parts (m)
# BF = distance of the fuselage over the wing (m)
# b = wing span (m)
# S = wing surface (m^2)
# phi_d = sweep angle (°)
# diedre_d = dihedral angle (°)
# n_ribs_1 = number of ribs (first sector)
# n_ribs_2 = number of ribs (second sector)

alpha = 2.5 
M_inf = 0.85
E = 70.*1e9
nu = 0.3
h_skins = np.array([0.031,0.028])
h_ribs = np.array([0.015,0.009])
h_spars_le = np.array([0.025,0.017])
h_spars_te = np.array([0.015,0.009])
BF = 5.96
b = np.array([64.75,75.65])
S = np.array([443.56,500.21])
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
VLM_geo = '../mesh/param_wing/VLM_mesh.geo'
a = airbus_wing()
span_1, span_2 =  a.compute_span()
L_1, c_1, c_2, c_3 = a.compute_cord()
e_1,e_2,e_3 = a.compute_ep() 
d_1 = d_2 = a.compute_diedre()
theta_1, theta_2, theta_3 = a.compute_twist()
phi_1, phi_2  = a.compute_phi()
coeff = 0.70
tck = np.array([[e_1,coeff*e_1],[e_2,coeff*e_2],[e_3,coeff*e_3]])
chord_pos = np.array([[0.10,0.8],[0.10,0.8],[0.10,0.8]])
create_VLM_mesh(span_1,span_2,theta_1,theta_2,theta_3,L_1,c_1,c_2,c_3,phi_1,phi_2,d_1,d_2)
create_FEM_mesh(VLM_geo,tck,chord_pos,n_ribs_1,n_ribs_2)
FEM_geo = '../mesh/param_wing/FEM_mesh.geo'
#Creating a file .msh
subprocess.call(['gmsh', VLM_geo, '-2'])
subprocess.call(['gmsh', FEM_geo, '-2'])
#Knowing the FEM nodes
f = open('../mesh/param_wing/FEM_mesh.msh','r')
lines = f.readlines()
FEM_nodes = int(lines[lines.index('$Nodes\n')+1])
f.close()

##COLLECTING SNAPSHOTS
#upgrate: for i_snap,(fspan,fsurf,...) in enumerate(itertools.product(b,S,h_skins...))
i_snap = 0
U = np.zeros((FEM_nodes*3,1))
Sigma = np.zeros((FEM_nodes*9,1))
for fspan in b: #Loop for wing span
    for fsurf in S: #Loop for wing surface
        for fskin in h_skins: #Loop for skin thickness
            for fribs in h_ribs: #Loop for rib thickness
                for fsple in h_spars_le: #Loop for spars LE thickness
                    for fspte in h_spars_te: #Loop for spars TE thickness
                        #1) Applying the right values in the wing
                        a_new = mesh_deformation_airbus('../mesh/param_wing/VLM_mesh.msh','../mesh/param_wing/FEM_mesh.msh') 
                        a_new.new_wing_mesh(fspan,fsurf,phi,diedre,BF,Mach = M_inf)
                        #2) Defining the files used and the parameters
                        vlm_mesh_file = '../mesh/param_wing/new_VLM.msh'
                        vlm_mesh_file_out = '../mesh/param_wing/new_VLM_def.msh'
                        fem_mesh = '../mesh/param_wing/new_FEM.msh'
                        param_data = np.zeros(9)
                        param_data[0] = alpha
                        param_data[1] = v_inf
                        param_data[2] = rho
                        param_data[3] = E
                        param_data[4] = nu
                        param_data[5] = fskin
                        param_data[6] = fribs
                        param_data[7] = fsple
                        param_data[8] = fspte
                        #3) Performing the analysis and saving the results
                        pw = PARAM_WING_VLM_FEM(param_data,i_snap)
                        U_it,Sigma_it = pw.do_analysis(vlm_mesh_file,vlm_mesh_file_out,fem_mesh)
                        U = np.hstack((U,U_it))
                        Sigma = np.hstack((Sigma,Sigma_it))
                        i_snap += 1

##LINKING i_snap WITH THE PARAMETERS VALUES
file_link = open(r'relationParam.txt','w')
file_link.write('RELATION OF i_snap WITH THE PARAMETERS VALUES\n')
file_link.write('\n')
file_link.write('i_snap'+"      "+"b"+"      "+"S"+"        "+"h_skin"+"      "+"h_ribs"+"     "+"h_spars_le"+"     "+"h_spars_te"+"\n")
j = 0
for fspan in b: #Loop for wing span
    for fsurf in S: #Loop for wing surface
        for fskin in h_skins: #Loop for skin thickness
            for fribs in h_ribs: #Loop for rib thickness
                for fsple in h_spars_le: #Loop for spars LE thickness
                    for fspte in h_spars_te: #Loop for spars TE thickness
                        file_link.write(str(j)+"        "+str(fspan)+"    "+str(fsurf)+"      "+str(fskin)+"      "+str(fribs)+"      "+str(fsple)+"      "+str(fspte)+"\n")
                        j += 1
file_link.close()

##SAVE DATA
np.save("U",U)
np.save("Sigma",Sigma)