function [R, q] = quasar(X, Y, sigma)

numOfPts = size(X,1);
Q = zeros(4*numOfPts+4);
% cbar2 = 1.;
noise_bound = chi2inv(1-1e-6,3)*sigma*sigma;
EYE4 = eye(4);

for k=1:numOfPts
    a = X(k,:);
    b = Y(k,:);
    
    RES = (a*a.'+b*b.')*EYE4 + 2*O12p([b,0], [a,0]);
    Q_kk = RES/noise_bound/2 + EYE4/2;
    Q_0k = RES/noise_bound/4 - EYE4/4;
    Q(k*4+(1:4),k*4+(1:4)) = Q_kk;
    Q(k*4+(1:4),(1:4)) = Q_0k; 
    Q((1:4),k*4+(1:4)) = Q_0k; 
end


%% solve SDP
cvx_begin quiet
    cvx_solver mosek%sdpt3
    variable Z(4*numOfPts+4, 4*numOfPts+4) symmetric
    minimize trace(Q*Z)
    subject to
    trace(Z(1:4,1:4)) == 1;
    for k = 1:numOfPts
        Z(k*4+(1:4),k*4+(1:4)) == Z(1:4,1:4);  
    end
    for k = 1:numOfPts+1
        for l = k+1:numOfPts+1
            %Z((k-1)*4+(1:4),(l-1)*4+(1:4)) == Z((k-1)*4+(1:4),(l-1)*4+(1:4)).'; 
            Z((k-1)*4+1,(l-1)*4+(2:4)) == Z((k-1)*4+(2:4),(l-1)*4+1).';
            Z((k-1)*4+2,(l-1)*4+(3:4)) == Z((k-1)*4+(3:4),(l-1)*4+2).';
            Z((k-1)*4+3,(l-1)*4+4)     == Z((k-1)*4+4,(l-1)*4+3).';
        end
    end
    Z == semidefinite(4*numOfPts+4);
cvx_end


%% eigenvalue decomposition
[V, D] = eig(Z); 
[~, idxmax] = max(diag(D)); 
q1 = V(:,idxmax);

R = quat2dcm([q1(4); q1(1:3)]'); 
q = quatnormalize([q1(4); q1(1:3)]');

