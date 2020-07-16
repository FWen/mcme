function [R, t] = mcme_6dof_regist(X, Y, sigma, q0, t0)

MAX_ITER = 10;
TOL = 1e-7;
TOL2 = 1e-5;
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
    t = zeros(3,1);
end

par.n = K + 1;
par.p = round(sqrt(2*par.n)/3);% parameter p for low-rank factor

% initialization of R
R0 = rand(par.n, par.p);
R0t = (R0 ./ repmat(sum(R0.^2,2).^0.5,1,par.p)).';

% parameters for L-BFGS
options.display = 'none';
options.optTol = 1e-8;
G_0k = zeros(4,K);

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
    
    for iter2=1:MAX_ITER
        qm1_2 = q;
        tm1_2 = t;
        
        % compute rotation q based on translation
        r1_ri = R(2:end,:)*R(1,:).';
        for k=1:K
            G_0k(:,k*4+(-3:0)) = O12p([Y(k,:) - t',0], [X(k,:),0]);
        end
        G = G_0k*kron(1-r1_ri,eye(4));

        [V, D] = eig(G);
        [~, idxmin] = min(diag(D));
        q = V(:,idxmin);

        % compute translation based on rotation q   
        Xh = O21p(q)*[X, zeros(K,1)]';
        t = (Y' - Xh(1:3,:))*(1-r1_ri) / sum(1-r1_ri);
        
        if norm(q-qm1_2,'fro')<TOL2 && norm(t-tm1_2,'fro')<TOL2
            break;
        end
    end
    
    if norm(q-qm1,'fro')<TOL && norm(t-tm1,'fro')<TOL
        break;
    end
end

R = quat2dcm([q(4);q(1:3)]');
