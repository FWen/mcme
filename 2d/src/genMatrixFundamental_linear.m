% Gen Quasiconvex Matrix for two set of keypoitns
% x1, x2: keypoints on image1 and image2 respectively
% A,b,c,d: Matrix for residual functions |Ax+b|_1/(cx+d)

function [A, b] = genMatrixFundamental_linear(X1, X2)

    nbpoints = size(X1,2);
    AA = [];
    for k=1:nbpoints
        AA = [AA; kron(X1(:,k),X2(:,k)).'];
    end    
    A = AA(:,1:8);
    b = -1*ones(nbpoints,1);   
end