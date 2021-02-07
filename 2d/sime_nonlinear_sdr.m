function [theta, inliers] = sime_nonlinear_sdr(A, b, c, w, sigma, theta0, th)

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

par.n = N + 1;
% parameter p for low-rank factor
par.p = round(sqrt(2*par.n)/3);

% initialization of R
R0 = rand(par.n, par.p);
R0t = (R0 ./ repmat(sum(R0.^2,2).^0.5,1,par.p)).';

% parameters for L-BFGS
options.display = 'none';
options.optTol = 1e-8;

Ax = A(1:2:end,:);
Ay = A(2:2:end,:);

par2.A = A; par2.b = b;
par2.c = c; par2.w = w;

for iter=1:MAX_ITER
    theta_m1 = theta;
    
    Q = zeros(N+1);
    Q(1,2:end) = beta - ((b(1,:)' + Ax*theta).^2 + (b(2,:)' + Ay*theta).^2)'./(theta'*c+w).^2;
    Q(2:end,1) = Q(1,2:end)';
        
    par.Q = Q;
    rr = minFunc(@objfun, R0t(:), options, par);
    R  = reshape(rr, par.p, par.n).';
    R = R ./ repmat(sum(R.^2,2).^0.5,1,par.p);
    
    s = 1 - R(2:end,:)*R(1,:).';
    par2.s = s;
    theta = minFunc(@objfun_quasiconvex, theta, options, par2);
   
    if norm(theta-theta_m1,'fro')<TOL
       break;
    end
end

res = ((b(1,:)' + Ax*theta).^2 + (b(2,:)' + Ay*theta).^2)'./(theta'*c+w).^2;
inliers = find(res<=beta);
