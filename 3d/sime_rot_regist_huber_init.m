function [R] = sime_rot_regist_huber_init(X, Y, sigma)

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

% Initialization
w = ones(1,K);
q = randn(4,1);
q = q/norm(q);
tao = sqrt(noise_bound);
for k=1:50
    qm1 = q;
    Gw = G_0k*kron(w.', eye(4));
    [V, D] = eig(Gw);
    [~, idxmin] = min(diag(D));
    q = V(:,idxmin);

    Xh = O21p(q)*[X, zeros(K,1)]';
    e = sum((Y' - Xh(1:3,:)).^2, 1).^0.5;
    w = tao./e;
    w(e<tao) = 1.;

    if norm(q-qm1,'fro')<TOL
        break;
    end
end

for iter=1:MAX_ITER
    qm1 = q;
    
    Xh = O21p(q)*[X, zeros(K,1)]';
    r1_ri = (beta - sum((Y' - Xh(1:3,:)).^2,1)/noise_bound)';
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
