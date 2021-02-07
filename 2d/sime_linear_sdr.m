function [theta, inliers, inliers_b] = sime_linear_sdr(A, y, sigma, theta0, th)

MAX_ITER = 10;
TOL = 1e-6;
[N, d] = size(A);

%initialization
if nargin>=4
    theta = theta0;
else
    theta = randn(d,1);
end

if nargin>=5
    beta = th^2; % noise bound 
else
    beta = chi2inv(1-1e-3,1)*sigma*sigma; % noise bound based on Chi-squared distribution test;
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

for iter=1:MAX_ITER
    theta_m1 = theta;
    
    Q = zeros(N+1);
    Q(1,2:end) = beta - ((y - A*theta).^2).';
    Q(2:end,1) = Q(1,2:end)';
        
    par.Q = Q;
    rr = minFunc(@objfun, R0t(:), options, par);
    R  = reshape(rr, par.p, par.n).';
    R = R ./ repmat(sum(R.^2,2).^0.5,1,par.p);
    
    s = 1 - R(2:end,:)*R(1,:).';
    As = repmat(s,1,d).*A;
    theta = inv(A'*As)*(As'*y);
   
    if norm(theta-theta_m1,'fro')<TOL
       break;
    end
end

res = (y - A*theta).^2;
inliers = find(res<=beta);
inliers_b = res<=beta;
