# Parametrized PDE
Author: Yaduo LI.

Advisors: Joseph MORLIER.

ISAE SUPAERO, 2019.

## Presentation
This project aims to presnet the comparison results of different mythologies for the parametrized PDE problem. The test is firstly taken on a Steady-Advection-Diffusion problem. The following methods are included:
  - Physics-Informed Neural Networks (PINN).
  - Reduced Order Models using Proper Orthogonal Decomposition (POD) method. The reduced base is computed using a greedy algorithm and also the TensorFlow package.
  - Gaussian Process to esitmate directly the computation results form a group of parameters. 
  - Kriging Model using the Partial Least Squares method (KPLS)

## Dependence
To luanch the testing notebook, the following should be used:
1. Python version 3.7 
2. The needed packages:
  - [pyDOE](https://anaconda.org/conda-forge/pydoe) and [Scikit learn](https://anaconda.org/anaconda/scikit-learn)
  - [deepxde](https://github.com/lululxvi/deepxde) for the PINN method
  - [TensorFlow](https://anaconda.org/conda-forge/tensorflow) for the TensorFlow based POD method
  - [SMT](https://github.com/jomorlier/SMT) for the KPLS method
  - [ipywidgets](https://anaconda.org/conda-forge/ipywidgets) for the visualization of the results
