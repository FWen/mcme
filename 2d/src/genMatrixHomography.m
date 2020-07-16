% Gen Quasiconvex Matrix for two set of keypoitns
% x1, x2: keypoints on image1 and image2 respectively
% A,b,c,d: Matrix for residual functions |Ax+b|_1/(cx+d)

function [A, b, c, d] = genMatrixHomography(X1, X2)

    nbpoints = size(X1,2);
    AA = [];
    bb = [];
    cc = [];
    dd = [];    
    for i=1:nbpoints
        x1 = X1(1,i); 
        y1 = X1(2,i);
        x2 = X2(1,i); 
        y2 = X2(2,i);
        
        ai1 = [x1 y1 1 0 0 0 -x2*x1 -x2*y1];
        ai2 = [0 0 0 x1 y1 1 -y2*x1 -y2*y1];
        AA = [AA; ai1; ai2];
        bb = [bb [-x2; -y2]];
        ci = [0 0 0 0 0 0 x1 y1];
        cc = [cc ci'];
        dd = [dd 1];               
    end    
    A = AA;
    b = bb;    
    c = cc;
    d = dd;
end