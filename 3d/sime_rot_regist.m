function [R] = sime_rot_regist(X, Y, sigma, q0)

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

for iter=1:MAX_ITER
    qm1 = q;
    
    Q = zeros(K+1);
    Xh = O21p(q)*[X, zeros(K,1)]';
    Q(1,2:end) = beta - sum((Y' - Xh(1:3,:)).^2,1)/noise_bound;

    r1_ri = Q(1,2:end)';
    indx1 = find(r1_ri>=0);
    indx2 = find(r1_ri<0);
    r1_ri(indx1) = -1;
    r1_ri(indx2) = 1;
    

    G = G_0k*kron(1-r1_ri, eye(4));
    
    [V, D] = eig(G);
    [~, idxmin] = min(diag(D));
    q = V(:,idxmin);
    
    if norm(q-qm1,'fro')<TOL
        break;
    end
end

R = quat2dcm([q(4);q(1:3)]');
