function [Inliers,theta] = homo_irw(A,b)

kap = 4;
m = size(A,1)/kap;
n = 8;

MAX_ITER = 10;
TOL = 1e-7;

C = [b; zeros(m,1)]; 
B = [sparse(n,1); ones(m,1)]; 
J = kron(speye(m),ones(kap,1));
A = [A, -J; sparse(m,n), -speye(m)];

K.l = size(A,1);
pars.eps = 1e-8;
pars.maxiter = 1e3;
pars.fid = 0;

[~,Y,~] = sedumi(A,-B,C,K,pars);
slack = Y(n+1:end);

for k=1:MAX_ITER-1
    slack_m1 = slack;
    
    w = (max(slack,0)+1e-3).^(-0.9);
    B = [sparse(n,1); w];
    [~,Y,~] = sedumi(A,-B,C,K,pars);
    slack = Y(n+1:end);
        
    if norm(slack-slack_m1)/sqrt(n)<TOL
        break
    end
end
Inliers = find(slack<1e-10);
theta = Y(1:n);
