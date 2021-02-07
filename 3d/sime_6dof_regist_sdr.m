function [R, t] = sime_6dof_regist_sdr(X, Y, sigma, q0, t0)

MAX_ITER = 10;
TOL = 1e-7;
K = size(X,1);
beta = 1.0;
noise_bound = chi2inv(1-1e-6,3)*sigma^2; % noise bound

% initialization
if nargin>=4
    q = [q0(2:4), q0(1)]';
else
    q = randn(4,1);
    q = q/norm(q);
end

if nargin>=5
    t = t0;
else
    t = (mean(Y,1)-mean(X,1))';
end

par.n = K + 1;
par.p = round(sqrt(2*par.n)/3); % parameter p for low-rank factor

% initialization of R
R0 = rand(par.n, par.p);
R0t = (R0 ./ repmat(sum(R0.^2,2).^0.5,1,par.p)).';

% parameters for L-BFGS
options.display = 'none';
options.optTol = 1e-8;

for iter=1:MAX_ITER
    qm1 = q;
    tm1 = t;
       
    Q = zeros(K+1);
    Xh = O21p(q)*[X, zeros(K,1)]';
    Q(1,2:end) = beta - sum((Y' - Xh(1:3,:) - repmat(t,1,K)).^2,1)/noise_bound;
    Q(2:end,1) = Q(1,2:end)';
        
    par.Q = Q;
    rr = minFunc(@objfun, R0t(:), options, par);
    R  = reshape(rr, par.p, par.n).';
    R = R ./ repmat(sum(R.^2,2).^0.5,1,par.p);
    r1_ri = R(2:end,:)*R(1,:).';
    
    %%
    w = 1 - r1_ri;
    ww = repmat(w,1,3);
    mu_x = sum(ww.*X,1)/sum(w);
    mu_y = sum(ww.*Y,1)/sum(w);
    Cxy = (ww.*(X-repmat(mu_x,K,1))).' * (Y-repmat(mu_y,K,1));
    A = Cxy - Cxy.';
    aa = [A(2,3), A(3,1), A(1,2)];
    Q = [trace(Cxy), aa; aa.', Cxy+Cxy.'-trace(Cxy)*eye(3)];
    [V, D] = eig(Q);
    [~, idxmin] = max(diag(D));
    q = [V(2:4,idxmin); V(1,idxmin)];
    
    xr = O21p(q)*[mu_x, 0]';
    t = mu_y' - xr(1:3);
        
    if norm(q-qm1,'fro')<TOL && norm(t-tm1,'fro')<TOL
        break;
    end
end

R = quat2dcm([q(4);q(1:3)]');
