function [score,vbInliers] = compute_homo_score(H21,X1,X2)

    N = size(X1,2);
    H12 = inv(H21);    
    
    % H21
    h11 = H21(1,1);    h12 = H21(1,2);    h13 = H21(1,3);
    h21 = H21(2,1);    h22 = H21(2,2);    h23 = H21(2,3);
    h31 = H21(3,1);    h32 = H21(3,2);    h33 = H21(3,3);
    
    % H12
    h11inv = H12(1,1);    h12inv = H12(1,2);    h13inv = H12(1,3);
    h21inv = H12(2,1);    h22inv = H12(2,2);    h23inv = H12(2,3);
    h31inv = H12(3,1);    h32inv = H12(3,2);    h33inv = H12(3,3);
   
    th=chi2inv(0.999,2); % the outlier rejection threshold
    currentScore = 0;

    vbCurrentInliers = zeros(N,1);
    for i=1:N
        bIn = true;

        u1 = X1(1,i);   v1 = X1(2,i);
        u2 = X2(1,i);   v2 = X2(2,i);

        % Reprojection error in first image // x2in1 = H12*x2
        w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        chiSquare1 = squareDist1*1;

        %currentScore = currentScore+ th - chiSquare1;
        if(chiSquare1>th)
            bIn = false;
        else
            currentScore = currentScore + th - chiSquare1;
        end
        
        % Reprojection error in second image    // x1in2 = H21*x1
        w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        chiSquare2 = squareDist2*1;

        %currentScore = currentScore+ th - chiSquare2;
        if(chiSquare2>th)
            bIn = false;
        else
            currentScore = currentScore+ th - chiSquare2;
        end

        if(bIn)
            vbCurrentInliers(i)=1;
        else
            vbCurrentInliers(i)=0;
        end

    end
    
    score = currentScore;
    vbInliers = find(vbCurrentInliers>0);
    %sum(vbCurrentInliers)
    