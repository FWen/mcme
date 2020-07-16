function [A, b] = genMatrix_Affinity(x1, x2)
    A = [];
    b = [];
    
    for k=1:size(x1,2)
        ai1 = [x2(:,k)',0,0,0];
        ai2 = [0,0,0,x2(:,k)'];
        A = [A; ai1; ai2];
        b = [b; x1(1,k); x1(2,k)];
    end
end