clear all; clc;

addpath src ../minFunc

N = 250;   % Number of points
d = 8;     % data dimension
sig = 0.1; % inlier std used to generate Gaussian noise
ons = 4;  % outlier noise strength

% Inlier threshold
inlier_th = sqrt(chi2inv(1-1e-3,1))*sig;

% parameters for the EP method, solver can be 'gurobi' or 'sedumi'
solver =  'gurobi'; EP_opt.lpsolver = prepareSolver(solver);EP_opt.solver = solver; 
EP_opt.maxAlpha = 1e10; EP_opt.QThresh = 1e-4; EP_opt.alpha = 0.5;   EP_opt.kappa = 5;   

out_ratio = 0:0.1:0.8;
for nr=1:length(out_ratio)
    parfor k=1:50
        [out_ratio(nr), k]

        %% generate data
        A  = 2*rand(N,d) - 1;
        xt = rand(d,1);
        y  = A*xt + sig*randn(N,1);
        n_outl = round(out_ratio(nr)*N);
        J = randperm(N);
        y(J(1:n_outl),:) = y(J(1:n_outl),:) + (2*rand(n_outl,1)-1)*ons; %uniformly distributed outliers
%        y(J(1:n_outl),:) = y(J(1:n_outl),:) + randn(n_outl,1)*ons/2; %Gaussian distributed outliers
        
        
        %% L1 method
        tic; 
        [l1Inliers, l1Theta] = l1_alg(A,y,inlier_th,1);
        Ai = A(l1Inliers,:);
        l1Theta_ref = inv(Ai.'*Ai)*Ai.'*y(l1Inliers);
        mc1(k) = numel(l1Inliers);
        Err1(k,nr) = norm(l1Theta-xt)/norm(xt);
        RefErr1(k,nr) = norm(l1Theta_ref-xt)/norm(xt);
        runtime1(k) = toc;

        
        %% Iteratively reweighted method
        tic; 
        [irwInliers, irwTheta] = irw_alg(A,y,inlier_th,1);
        Ai = A(irwInliers,:);
        irwTheta_ref = inv(Ai.'*Ai)*Ai.'*y(irwInliers);        
        mc2(k) = numel(irwInliers);
        Err2(k,nr) = norm(irwTheta-xt)/norm(xt);
        RefErr2(k,nr) = norm(irwTheta_ref-xt)/norm(xt);
        runtime2(k) = toc;
               
        
        %%  RANSAC           
        [rsTheta, rsInliers, rsRuntime ] = linearFit(A, y, inlier_th, 'RANSAC', randn(1,d));
        Ai = A(rsInliers,:);
        rsTheta_ref = inv(Ai.'*Ai)*Ai.'*y(rsInliers);        
        mc3(k) = numel(rsInliers);
        Err3(k,nr) = norm(rsTheta-xt)/norm(xt);
        RefErr3(k,nr) = norm(rsTheta_ref-xt)/norm(xt);
        runtime3(k) = rsRuntime;

        
        %%  EP
        [epTheta, epInliers, epRuntime] = linearFit(A, y, inlier_th, 'EP', l1Theta_ref, EP_opt);
        Ai = A(epInliers,:);
        epTheta_ref = inv(Ai.'*Ai)*Ai.'*y(epInliers);    
        mc4(k) = numel(epInliers);
        Err4(k,nr) = norm(epTheta-xt)/norm(xt);
        RefErr4(k,nr) = norm(epTheta_ref-xt)/norm(xt);
        runtime4(k) = runtime1(k)+epRuntime;
        
                
        %% AM
        tic
        [mmTheta, mmInliers] = sime_linear(A, y, sig, l1Theta_ref);
        mc5(k) = numel(mmInliers);
        Err5(k,nr) = norm(mmTheta-xt)/norm(xt);
        runtime5(k) = runtime1(k)+toc;
        
        %% AM-R
        tic
        [mmTheta, mmInliers] = sime_linear_sdr(A, y, sig, l1Theta_ref);
        mc6(k) = numel(mmInliers);
        Err6(k,nr) = norm(mmTheta-xt)/norm(xt);
        runtime6(k) = runtime1(k)+toc;
        
    end
    av_mc(:,nr) = mean([mc1; mc2; mc3; mc4; mc5; mc6], 2);
    av_rt(:,nr) = mean([runtime1; runtime2; runtime3; runtime4; runtime5; runtime6], 2);
end

figure(1);
plot(out_ratio,av_mc(3,:),'r--',out_ratio,av_mc(1,:),'g--+',out_ratio,av_mc(2,:),'c--*',...
    out_ratio,av_mc(4,:),'k--^',out_ratio,av_mc(5,:),'b--o',out_ratio,av_mc(6,:),'m--d','linewidth',1); 
xlabel('Outlier ratio'); grid on;ylabel('Consensus Size')
legend('RANSAC','L1','IRW','EP','AM','AM-R', 'Location', 'Best');

figure(2);
semilogy(out_ratio,av_rt(3,:),'r--',out_ratio,av_rt(1,:),'g--+',out_ratio,av_rt(2,:),'c--*',...
    out_ratio,av_rt(4,:),'k--^',out_ratio,av_rt(5,:),'b--o',out_ratio,av_rt(6,:),'m--d','linewidth',1); 
xlabel('Outlier ratio'); grid on;ylabel('Runtime (seconds)')
legend('RANSAC','L1','IRW','EP','AM','AM-R', 'Location', 'Best');

f=figure(3);
m=10;
x = [Err3; RefErr3; Err1; RefErr1; Err2; RefErr2; Err4; RefErr4; Err5; Err6];
g1 = kron([1:m]', ones(size(Err1,1),nr)); g1 = g1(:);
g2 = repmat(1:nr,size(x,1),1); g2 = g2(:);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[6, 1],'symbol','.','outliersize',4,'widths',0.9,...
    'factorseparator',[1],'colors',hsv(m+1));
xlabel('Outlier ratio'); ylabel('Model estimation error'); set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5);% ylim([1e-3,3e0]);
set(findobj(gca,'tag','Outliers'),'Marker','+');  
for k=1:nr
    annotation(f,'textbox',[((k-1)+1.5)/(nr+2), 0.11, 0.035, 0.075],...
    'String',{out_ratio(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'RANSAC','RANSAC-r','L1','L1-r','IRW','IRW-r',...
    'EP','EP-r','AM','AM-R'}, 'Location', 'Best');
