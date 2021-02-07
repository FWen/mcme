function [theta, inliers, inliers_b] = sime_linear(A, y, sigma, theta0, th)

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

for iter=1:MAX_ITER
    theta_m1 = theta;
    
    r1_ri = beta - ((y - A*theta).^2);
    indx1 = find(r1_ri>=0);
    indx2 = find(r1_ri<0);
    r1_ri(indx1) = -1;
    r1_ri(indx2) = 1;
    
    As = repmat(1-r1_ri,1,d).*A;
    theta = inv(A'*As)*(As'*y);
   
    if norm(theta-theta_m1,'fro')<TOL
       break;
    end
end

res = (y - A*theta).^2;
inliers = find(res<=beta);
inliers_b = res<=beta;
