%% Generate random data
%  N: number of data points
%  d: dimensions
function [A,y,xt] = genRandomLinearData2(N, dim, in_var, out_var, outlierP)
 
%     if (nargin < 6)
%         balance = 1;
%     end
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

    % outliers with strong hypothese, 50% of the outliers
    m2 = rand(n-1, 1);    
    c2 = randn; 
    xo = -1 + 2.*rand(n-1,N);  %
    yo = m2'*xo + repmat(c2,1,N); % b0 = Ao*xo
    yo = yo + sig*randn(1,N);     % bo = Ao*xo
    
    
    A = [x.', ones(N, 1)];
    
    for k=1:t      
        if k<=t*0.8
            y(k) = y(k)+ osig*(rand-0.5);
        else
            y(k) = yo(k);
            A(k,:) = [xo(:,k).',1];
        end
    end
    
    y = y';
    xt = [m;c];

end
