function [R, t] = sime_6dof_regist(X, Y, sigma, q0, t0)

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

for iter=1:MAX_ITER
    qm1 = q;
    tm1 = t;
       
    Xh = O21p(q)*[X, zeros(K,1)]';
    r1_ri = beta - sum((Y' - Xh(1:3,:) - repmat(t,1,K)).^2,1)'/noise_bound;
    indx1 = find(r1_ri>=0);
    
    %%
    if isempty(indx1)
        mu_x = mean(X,1);
        mu_y = mean(Y,1);
        Cxy = ((X-repmat(mu_x,K,1))).' * (Y-repmat(mu_y,K,1));
    else
        mu_x = mean(X(indx1,:),1);
        mu_y = mean(Y(indx1,:),1);
        Cxy = X(indx1,:).'*Y(indx1,:)/length(indx1)-mu_x.'*mu_y;
    end
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
