%QUASICONVEX
function [theta, bestInls, iter, hist] = maxcon_consensus_ADMM_quasiconvex(z0, A, b, rho, maxIter, AA, bb, cc, dd, th, config)    
    

    [Alsv, blsv] = genQPLinearConstraints_sv(A, b);    
    N = size(A, 1);  dimension = size(AA,2);                
    C = [A -A*ones(dimension,1)];
    % Sanity check for initial point
    tt = z0;       
    rtheta = tt(end-dimension:end-1) - repmat(tt(end),dimension,1);                                   
    [~, ~, initinls] = compute_residuals_l1(AA, bb, cc, dd, rtheta, th);    
    %disp (length(initinls));          
    % Initialization for best results    
    bestInls = initinls;
    maxInls = length(initinls);
    bestTheta = rtheta;
    
    %Consensus ADMM initialization:    
    
    bestz = z0;    
    hist = [];
    %Consensus ADMM initialization:    
    sv = bestz(N+1:end); 
    u = bestz(1:N); s = sv(1:N); v = sv(N+1:end);
    ui = u; si = s;  sc = s; 
    vc = v;  vi = repmat(v, 1, N);   
    lambdaU = zeros(N, 1);
    lambdaS = zeros(N,1);  lambdaSC = lambdaS;
    lambdaV = zeros(dimension+1, N);  lambdaVC = zeros(size(v));          
    Aq = eye(N+dimension+1);
    Aqi_list = cell(N,1);
    for i=1:N
        Aqi = eye(dimension+1);
        rowi = C(i,:);
        for j=1:dimension+1
            Aqi(j,j) = rowi(j)^2 + 1;
            for k=j+1:dimension+1
                Aqi(j,k)=rowi(j)*rowi(k); Aqi(k,j) = rowi(j)*rowi(k);                
            end
        end        
        Aqi_list{i} = Aqi;
    end                
    % Start the ADMM iterations:
    iter = 0;    
    svi_diff = zeros(N,1);    
    %Lp = computeLarangian;
    while (iter < maxIter)
        iter = iter + 1;                
        if (iter > 1 && rho < config.max_rho); rho = rho*config.ADMM_rho_increase_rate; end                      
        %z^i update:        
        sld = s - lambdaS;                
        for i=1:N            
            vld = v - lambdaV(:,i);                        
            if (abs(sld(i)) > 0)
                vi(:,i) = vld; ui(i) = 0; si(i) = 0; obj1 = rho*(sld(i)^2 + u(i)^2);                     
                
                Aqi = Aqi_list{i};
                bqi = -2.*((b(i)+sld(i)).*C(i,:)' + vld);                                                      
                vvi = mldivide(-2.*Aqi, bqi);               

               uii = 1;  sii = C(i,:)*vvi - b(i); 
               obj2 = uii  + rho*((uii - u(i))^2 +(sii - sld(i))^2 + norm(vvi - vld)^2);               
               if obj2 < obj1
                    vi(:,i) = vvi; si(i) = sii; ui(i) = uii;                
               end                            
               
            else
                ui(i) = 0; si(i) = 0; vi(:,i) = vld; 
            end
            svi_diff(i) = norm([si(i); vi(:,i)] - [s(i); v])^2;                        
        end        
        %------------------sc, vc update---------------------------------                        
        bq = -2.*[(s - lambdaSC); (v - lambdaVC)];
        [scvc]= myQuadProg(Aq, bq, Alsv, blsv, 1e-9, -bq./2,'>', config.solver); 
        sc = scvc(1:N); 
        vc = scvc(N+1:end);        
        %------------u, s, v update---------------------------------------        
        pusv = [u; s; v];               
        u = (rho*1.0/(rho+1))*(ui + lambdaU);
        s = 0.5*((si + lambdaS) + (sc + lambdaSC));
        v = sum([vi + lambdaV  vc + lambdaVC], 2)./((N+1)*1.0);                                         
        zdiff = norm(pusv - [u; s;v]).^2;        
        
        % lambda update            
        lambdaU = lambdaU + ui - u;        
        lambdaS = lambdaS +  si - s;
        lambdaV = lambdaV + vi - repmat(v, 1, N);                 
        lambdaSC = lambdaSC + sc - s;
        lambdaVC = lambdaVC + vc - v;
        

        zz = [vc vi v];        
        
        % Save the best results. 
        for zidx=1:size(zz,2)
            tt = zz(:,zidx);
            rtheta = tt(end-dimension:end-1) - repmat(tt(end),dimension,1);      
            [~, ~, inls]=compute_residuals_l1(AA, bb, cc, dd, rtheta, th);                                     
            
            if (length(inls) > maxInls)            
                bestInls = inls;
                maxInls = length(inls);
                bestTheta = rtheta;                                               
            end        
        end                        
        disp(['Iter = ' num2str(iter) ' # Inliers = ' num2str(maxInls) ...
              '  zdiff = ' num2str(zdiff) ...
                            
              ]);
        
        % Stopping criteria
        if (iter>5 && zdiff < config.ADMM_zdiff)
            break; 
       end                            
        
    end        
    disp(['Max Inls = ' num2str(maxInls)]);
    theta = bestTheta;     
    %-------------------------------------------------------------------
end




%vvi = gurobiQuadProg(Aqi, bqi,zeros(1,dimension+1),0, 1e-9,vld,'=');
%vvi = quadprog(2.*Aqi, bqi);
%vvi = (-2.*Aqi)\(bqi);                     

%          if (abs(min(Alsv*(-bq./2) + blsv))<=1e-5)
%              scvc = -bq./2;
%          else
%              scvc= gurobiQuadProg(Aq, bq, Alsv, blsv, 1e-9, sv); 
%          end
                
%[xbz, ybz]= genStartingPointADMM(A, b, rtheta);
                %bestz = [xbz; ybz];
                %hist.bestTime = toc(startTime);
                %bestzUpdated = true;                
                
% Case 2: si > 0; ui = 1; Solve quadratic programming
%svi = gurobiQuadProg(eye(dimension+2), -2.*[sp(i); vld], [1 -C(i,:)], b(i), 1e-9, [sp(i);vld], '=');   