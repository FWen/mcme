# SIME
Simultaneous Inlier Identification and Model Fitting Using Truncated Loss

This is the code for reproducing the results of the MCME algorithm in the paper "Fei Wen, H. Wei, Y. Liu, P. Liu, Simultaneous Inlier Identification and Model Fitting Using Truncated Loss, 2020"

Note that, some codes of Huu Le (https://www.researchgate.net/publication/320707327demo_pami, see the 2d/src folder, including the RANSAC and EP methods and their dependency), Alvaro Parra Bustos (https://cs.adelaide.edu.au/∼aparra/project/gore/sup.zip, see the 3d and 3d/common folders, including the ransac3dof, ransac6dof, and gore6 methods), and M. Schmidt (minFunc: unconstrained differentiable multivariate optimization in Matlab, http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005), which are used in our paper, are directly coped here to facilitate the ease of use for interested readers. They are copied here only for academic use purpose.

In the 2D registration experiment, SeDuMi is needed for the L1 and IRW methods, and SeDuMi or Gurobi is needed for the EP method.
