function [H,Theta_ref] = linear_model_refine(A,b,Inliers,data)

indx = kron(Inliers,[2,2]) - kron(ones(size(Inliers)),[1,0]);
Ai = A(indx,:);
Theta_ref = inv(Ai.'*Ai)*Ai.'*b(indx);
H = inv(data.T2)*reshape([Theta_ref; 1], [3,3])'*data.T1;
