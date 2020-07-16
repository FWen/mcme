%% Generate random data
%  N: number of data points
%  d: dimensions
function [A,y,xt] = genRandomLinearData3(N, dim, in_var, out_var, outlierP)
 
    rng('shuffle');
   
    sig = in_var;   % Inlier Varience
    osig = out_var; % Outlier Varience
    n = dim;        % Dimension of space
    m = rand(n-1, 1);    
    c = randn; 
    
    %% Generate data
    x = -1 + 2.*rand(n-1,N);  %A
    y = m'*x + repmat(c,1,N); % b = A*x

    
    %% Perturb data by Gaussian noise
    gNoise = sig*randn(1,N);
    y = y + gNoise; % b = A*x + n
    
    %% Generate outliers   
    t = outlierP; 
    t = round(N*t/100); 

    % outliers with strong hypothese, 35% of the outliers
    m2 = rand(n-1, 1);
    c2 = randn; 
    x2 = -1 + 2.*rand(n-1,N);  %A2
    y2 = m2'*x2 + repmat(c2,1,N); % b0 = Ao*xo
    y2 = y2 + sig*randn(1,N);     % bo = Ao*xo
    
    m3 = rand(n-1, 1);
    c3 = randn; 
    x3 = -1 + 2.*rand(n-1,N);  %A3
    y3 = m3'*x3 + repmat(c3,1,N); % b0 = Ao*xo
    y3 = y3 + sig*randn(1,N);     % bo = Ao*xo
    
    A = [x.', ones(N, 1)];
    
    for k=1:t      
        if k<=t*0.3
            y(k) = y(k)+ osig*(rand-0.5);
        elseif k<=t*0.65 & k>t*0.3
            y(k) = y2(k);
            A(k,:) = [x2(:,k).',1];
        else
            y(k) = y3(k);
            A(k,:) = [x3(:,k).',1];
        end
    end
    
    y = y';
    xt = [m;c];
end
