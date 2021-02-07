function [R] = sime_rot_regist_sdr(X, Y, sigma, q0)

MAX_ITER = 10;
TOL = 1e-7;
K = size(X,1);
beta = 1.0;
noise_bound = chi2inv(1-1e-6,3)*sigma*sigma; % noise bound

G_0k = zeros(4,K);
for k=1:K
    a = X(k,:); b = Y(k,:);
    G_0k(:,k*4+(-3:0)) = O12p([b,0], [a,0]);
end

%initialization
if nargin>=4
    q = [q0(2:4), q0(1)]';
else
    q = randn(4,1);
    q = q/norm(q);
end

par.n = K + 1;
% parameter p for low-rank factor
par.p = round(sqrt(2*par.n)/3);

% initialization of R
R0 = rand(par.n, par.p);
R0t = (R0 ./ repmat(sum(R0.^2,2).^0.5,1,par.p)).';

% parameters for L-BFGS
options.display = 'none';
options.optTol = 1e-8;

for iter=1:MAX_ITER
    qm1 = q;
    
    Q = zeros(K+1);
    Xh = O21p(q)*[X, zeros(K,1)]';
    Q(1,2:end) = beta - sum((Y' - Xh(1:3,:)).^2,1)/noise_bound;
    Q(2:end,1) = Q(1,2:end)';
        
    par.Q = Q;
    rr = minFunc(@objfun, R0t(:), options, par);
    R  = reshape(rr, par.p, par.n).';
    R = R ./ repmat(sum(R.^2,2).^0.5,1,par.p);
    
    r1_ri = R(2:end,:)*R(1,:).';

    G = G_0k*kron(1-r1_ri,eye(4));
    
    [V, D] = eig(G);
    [~, idxmin] = min(diag(D));
    q = V(:,idxmin);
    
    if norm(q-qm1,'fro')<TOL
        break;
    end
end

R = quat2dcm([q(4);q(1:3)]');
