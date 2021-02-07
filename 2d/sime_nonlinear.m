function [theta, inliers] = sime_nonlinear(A, b, c, w, sigma, theta0, th)

MAX_ITER = 10;
TOL = 1e-6;
[d,N] = size(c);

%initialization
if nargin>=4
    theta = theta0;
else
    theta = randn(d,1);
end

if nargin>=7
    beta = th^2; % noise bound 
else
    beta = chi2inv(1-1e-3,1)*sigma*sigma; % noise bound based on Chi-squared distribution test
end

% parameters for L-BFGS
options.display = 'none';
options.optTol = 1e-8;

Ax = A(1:2:end,:);
Ay = A(2:2:end,:);

par2.A = A; par2.b = b;
par2.c = c; par2.w = w;

for iter=1:MAX_ITER
    theta_m1 = theta;
    
    r1_ri = beta - ((b(1,:)' + Ax*theta).^2 + (b(2,:)' + Ay*theta).^2)'./(theta'*c+w).^2;
    indx1 = find(r1_ri>=0);
    indx2 = find(r1_ri<0);
    r1_ri(indx1) = -1;
    r1_ri(indx2) = 1;

    par2.s = 1 - r1_ri';
    theta = minFunc(@objfun_quasiconvex, theta, options, par2);
   
    if norm(theta-theta_m1,'fro')<TOL
       break;
    end
end

res = ((b(1,:)' + Ax*theta).^2 + (b(2,:)' + Ay*theta).^2)'./(theta'*c+w).^2;
inliers = find(res<=beta);
