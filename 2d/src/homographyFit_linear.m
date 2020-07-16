% Perform homography estimation between two images
% data.x1: Keypoints found on image 1
% data.x2: Keypoints found on image 2
% method: 'RANSAC' for ransac and 'EP' for Exact Penalty method

function [theta, inls, runtime] = homographyFit_linear(data, th, method, theta0, config)

    x1 = data.x1;
    x2 = data.x2;        
    th = th*data.T2(1,1); % The data was normalized, 
                         % the inlier threshold (in pixel) should be normalized         
    
    [A, b] = genMatrixHomography_linear(x1, x2);
    [lA, lb] = genLinearMatrixFromLinear(A, b, th);    
    
    if (strcmp(method,'RANSAC'))                      
        tic    
        [ransacH] =  ransacfithomography(x1, x2, th);                
        runtime = toc;       
        theta = ransacH(:);
                    
    elseif (strcmp(method,'EP'))

        alpha = config.alpha;
        kappa = config.kappa;        
        QThresh = config.QThresh;
        alphaMax = config.alphaMax;  
        
        % Normalize matrix if a 3x3 matrix was supplied
        if (length(theta0)==9)
            theta0 = theta0./theta0(end);             
            theta0 = theta0(1:end-1);
        end                   
        [x0, y0] = genStartingPoint(lA, lb, theta0);        % From theta0, compute u0,s0, v0
        tic
        while (true)
            % Execute Frank-Wolfe algorithm
            [x0, y0, theta, P, F, Q] = fwQuasiconvex(lA, lb, c, d, x0, y0, alpha, config);

%             printSeparator('-');
            if ( Q <= QThresh || alpha > alphaMax)                   % Reach feasible region, stop
%                 disp(['EP TERMINATED as Q(z) reaches ' num2str(Q)]);                
                break;
            end
            alpha = kappa*alpha;                                     % increase alpha
            
        end        
        runtime = toc;          
        
    elseif (strcmp(method, 'ADMM'))        
         %Normalize theta0 to make H33 = 1;      
        if (length(theta0)==9)
            theta0 = theta0./theta0(end);             
            theta0 = theta0(1:end-1);
        end    
        tic
        [x0, y0] = genStartingPoint(lA, lb, theta0);
        [theta, ~, ~, ~] = maxcon_consensus_ADMM_quasiconvex([x0;y0], lA, lb, config.rho_admm, config.ADMM_max_iter, A, b, c, d, th, config);
        runtime = toc;
    else 
        error('Unkown method');        
        
    end    
    
    [~, ~, inls]=compute_residuals_l1(A, b, c, d, theta, th);

       
end