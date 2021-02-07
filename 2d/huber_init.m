function [theta] = huber_init(A, y, sigma)

MAX_ITER = 50;
TOL = 1e-6;
[N, d] = size(A);

theta = randn(d,1);
beta = chi2inv(1-1e-3,1)*sigma*sigma; % noise bound based on Chi-squared distribution test;
w = ones(N,1);
tao = sqrt(beta);
for k=1:MAX_ITER
    thetam1 = theta;
    
    As = repmat(w,1,d).*A;
    theta = inv(A'*As)*(As'*y);
    
    e = abs(y - A*theta);
    w = tao./e;
    w(e<tao) = 1.;

%   residual(k) = norm(q-qm1,'fro');
    if norm(theta-thetam1,'fro')<TOL
        break;
    end
end
