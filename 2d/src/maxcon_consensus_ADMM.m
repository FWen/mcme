% New formulation for ADMM to solve maximum consensus problem using
% consensus form of ADMM
function [theta, bestInls, iter, hist] = maxcon_consensus_ADMM(z0, A, b, rho, maxIter, x, y, th, config)        
    % Prepare the matrices
    [Alsv, blsv] = genQPLinearConstraints_sv(A, b);    
    N = 2*size(x, 1);   dimension = size(x,2);              
    C = [A -A*ones(dimension,1)];
    
    % Get initial inliers
    tt = z0;   
    rtheta = tt(end-dimension:end-1) - repmat(tt(end),dimension,1);                               
    initinls = find(abs(y-x*rtheta)<=th);  disp (length(initinls));          
    
    % Initialization for best results    
    bestInls = initinls;
    maxInls = length(initinls);
    bestTheta = rtheta;    
    
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
    
    % Prepare the matrices for optimization    
    d1 = dimension+1;
    Aq = eye(N+d1);
    Aqi_list = cell(N,1);    
    for i=1:N
        Aqi = eye(d1);
        cc = C(i,:);
        for j=1:d1
            Aqi(j,j) = cc(j)^2 + 1;
            for k=j+1:d1
                Aqi(j,k)=cc(j)*cc(k); Aqi(k,j) = cc(j)*cc(k);                
            end
        end        
        Aqi_list{i} = Aqi;                
    end            
    
    % Start the ADMM iterations:
    iter = 0;    
    svi_diff = zeros(N,1);        
    
    while (iter < maxIter)
        iter = iter + 1;                
        if (iter > 1 && rho < config.max_rho); rho = rho*config.ADMM_rho_increase_rate; end                                      
        %zi update:        
        sld = s - lambdaS; 
        ui = zeros(N,1); si = zeros(N,1); vi = repmat(v,1,N) - lambdaV;        
        
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
        scvc = myQuadProg(Aq, bq, Alsv, blsv, 1e-9, -bq./2,'>');
        sc = scvc(1:N); vc = scvc(N+1:end);              
        
        %------------u, s, v update---------------------------------------
        pusv = [u; s; v];               
        u = (rho*1.0/(rho+1))*(ui + lambdaU);
        s = 0.5*((si + lambdaS) + (sc + lambdaSC));
        v = sum([vi + lambdaV  vc + lambdaVC], 2)./((N+1)*1.0);                                         
        
        usvdiff = norm(pusv - [u; s;v]);        
        
        
        % u update            
        lambdaU = lambdaU + ui - u;        
        lambdaS = lambdaS +  si - s;
        lambdaV = lambdaV + vi - repmat(v, 1, N);                 
        lambdaSC = lambdaSC + sc - s;
        lambdaVC = lambdaVC + vc - v;
                
        zz = [vi vc v];              
        for zidx=1:size(zz,2)
            tt = zz(:,zidx);
            rtheta = tt(end-dimension:end-1) - repmat(tt(end),dimension,1);      
            inls = find(abs(y-x*rtheta)<=th);                                                 
            if (length(inls) > maxInls)            
                bestInls = inls;
                maxInls = length(inls);
                bestTheta = rtheta;                               
            end        
        end                        
        
        % --------------- DEBUG ----------------------------------------%        
        disp(['Iter = ' num2str(iter)  ' # Inliers = ' num2str(maxInls) ...
              '  z_diff = ' num2str(usvdiff)]);  ...                            
        %------------------Stopping criteria-----------------------------
        if (iter > 5 && ( usvdiff < config.ADMM_zdiff ))
             break; 
        end                                    
    end        
    disp(['Max Inls = ' num2str(maxInls)]);
    theta = bestTheta;     
    %-------------------------------------------------------------------      
end

