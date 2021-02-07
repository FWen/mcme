function [Inliers,theta] = homo_L1(A,b)

kap = 4;
m = size(A,1)/kap;
n = 8;

C = [b; sparse(kap*m,1)]; 
B = [sparse(n,1); ones(kap*m,1)]; 
A = [A, -speye(kap*m); 
    sparse(kap*m,n), -speye(kap*m)];

K.l = size(A,1);
pars.eps = 1e-8;
pars.maxiter = 1e3;
pars.fid = 0;

[~,Y,~] = sedumi(A,-B,C,K,pars);
slack = sum(reshape(Y(n+1:end),kap,m));
Inliers = find(slack<1e-7)';
theta = Y(1:n);
