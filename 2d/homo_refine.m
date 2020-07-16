function [H,Theta_ref] = homo_refine(A,b,Inliers,data)

indx = kron(Inliers,4*ones(4,1)) - kron(ones(size(Inliers)),[3:-1:0]');
Ai = A(indx,:);
Theta_ref = inv(Ai.'*Ai)*Ai.'*b(indx);
H = inv(data.T2)*reshape([Theta_ref; 1], [3,3])'*data.T1;

    