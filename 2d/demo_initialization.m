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

outlier_ratio = 0.1:0.1:0.8;
for nr=1:length(outlier_ratio)
    parfor k=1:50
        [outlier_ratio(nr), k]

        %% generate data
        A  = 2*rand(N,d) - 1;
        xt = rand(d,1);
        y  = A*xt + sig*randn(N,1);
        n_outl = round(outlier_ratio(nr)*N);
        J = randperm(N);
        y(J(1:n_outl),:) = y(J(1:n_outl),:) + (2*rand(n_outl,1)-1)*ons; %uniformly distributed outliers
%        y(J(1:n_outl),:) = y(J(1:n_outl),:) + randn(n_outl,1)*ons/2; %Gaussian distributed outliers
        
        
        %% L1 method
        tic; 
        [l1Inliers, l1Theta] = l1_alg(A,y,inlier_th,1);
        Ai = A(l1Inliers,:);
        l1Theta_ref = inv(Ai.'*Ai)*Ai.'*y(l1Inliers);
        ErrL1(k,nr) = norm(l1Theta-xt)/norm(xt);
        RefErrL1(k,nr) = norm(l1Theta_ref-xt)/norm(xt);
        L1Runtime = toc; 
         
        
        %%  RANSAC 
        [rsTheta, rsInliers, rsRuntime ] = linearFit(A, y, inlier_th, 'RANSAC', randn(1,d));
        Ai = A(rsInliers,:);
        rsTheta_ref = inv(Ai.'*Ai)*Ai.'*y(rsInliers);
        ErrRs(k,nr) = norm(rsTheta-xt)/norm(xt);
        RefErrRs(k,nr) = norm(rsTheta_ref-xt)/norm(xt);


        %%  Huber  
        tic;
        huberTheta = huber_init(A, y, sig);
        huberRuntime = toc;
        
         %% AM - wo initialization
         tic;
        [mmTheta1, mmInliers1] = sime_linear(A, y, sig);
        mc1(k) = numel(mmInliers1);
        Err1(k,nr) = norm(mmTheta1-xt)/norm(xt);
        runtime1(k) = toc;
       
         %% AM - L1 initialization
         tic;
        [mmTheta2, mmInliers2] = sime_linear(A, y, sig, l1Theta_ref);
        mc2(k) = numel(mmInliers2);
        Err2(k,nr) = norm(mmTheta2-xt)/norm(xt);
        runtime2(k) = toc  + L1Runtime;

         %% AM - Huber initialization
         tic;
        [mmTheta3, mmInliers3] = sime_linear(A, y, sig, huberTheta);
        mc3(k) = numel(mmInliers3);
        Err3(k,nr) = norm(mmTheta3-xt)/norm(xt);
        runtime3(k) = toc + huberRuntime;
        
         %% AM - RANSAC initialization
         tic;
        [mmTheta4, mmInliers4] = sime_linear(A, y, sig, rsTheta_ref);
        mc4(k) = numel(mmInliers4);
        Err4(k,nr) = norm(mmTheta4-xt)/norm(xt);
        runtime4(k) = toc + rsRuntime;
        
         %% AM-R - wo initialization
         tic;
        [mmTheta5, mmInliers5] = sime_linear_sdr(A, y, sig);
        mc5(k) = numel(mmInliers5);
        Err5(k,nr) = norm(mmTheta5-xt)/norm(xt);
        runtime5(k) = toc;
        
         %% AM-R - L1 initialization
         tic;
        [mmTheta6, mmInliers6] = sime_linear_sdr(A, y, sig, l1Theta_ref);
        mc6(k) = numel(mmInliers6);
        Err6(k,nr) = norm(mmTheta6-xt)/norm(xt);
        runtime6(k) = toc + L1Runtime;
        
         %% AM-R - Huber initialization
         tic;
        [mmTheta7, mmInliers7] = sime_linear_sdr(A, y, sig, huberTheta);
        mc7(k) = numel(mmInliers7);
        Err7(k,nr) = norm(mmTheta7-xt)/norm(xt);
        runtime7(k) = toc + huberRuntime;

         %% AM-R - RANSAC initialization
         tic;
        [mmTheta8, mmInliers8] = sime_linear_sdr(A, y, sig, rsTheta_ref);
        mc8(k) = numel(mmInliers8);
        Err8(k,nr) = norm(mmTheta8-xt)/norm(xt);
        runtime8(k) = toc + rsRuntime;
    end
    av_mc(:,nr) = mean([mc1; mc2; mc3; mc4; mc5; mc6; mc7; mc8], 2)
    av_rt(:,nr) = mean([runtime1; runtime2; runtime3; runtime4; runtime5; runtime6; runtime7; runtime8], 2)
end

%% Model estimation error
m=2;
x = [Err1; Err5];
g1 = kron([1:m]', ones(size(Err1))); g1 = g1(:);
g2 = repmat(1:nr,size(x,1),1); g2 = g2(:);

f=figure(1);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Model estimation error');ylim([7e-3,5]);
set(gca,'yscale','log');grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(Err1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(Err1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratio(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'AM (Random init.)','AM-R (Random init.)'}, 'Location', 'Best');


%%
x = [Err3; Err7];
g1 = kron([1:m]', ones(size(Err1))); g1 = g1(:);
g2 = repmat(1:nr,size(x,1),1); g2 = g2(:);

f=figure(2);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Model estimation error');ylim([7e-3,5]);
set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(Err1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(Err1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratio(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '})
box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'AM (Huber init.)','AM-R (Huber init.)'}, 'Location', 'Best');


%%
x = [Err2; Err6];
g1 = kron([1:m]', ones(size(Err1))); g1 = g1(:);
g2 = repmat(1:nr,size(x,1),1); g2 = g2(:);

f=figure(3);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Model estimation error');ylim([7e-3,5]);
set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(Err1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(Err1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratio(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '})
box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'AM (L1 init.)','AM-R (L1 init.)'}, 'Location', 'Best');


%%
x = [Err4; Err8];
g1 = kron([1:m]', ones(size(Err1))); g1 = g1(:);
g2 = repmat(1:nr,size(x,1),1); g2 = g2(:);

f=figure(4);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Model estimation error');ylim([7e-3,5]);
set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(Err1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(Err1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratio(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'AM (RANSAC init.)','AM-R (RANSAC init.)'}, 'Location', 'Best');


%% runtime
figure(12);
semilogy(outlier_ratio,av_rt(1,:),'r--+',outlier_ratio,av_rt(2,:),'r--^',outlier_ratio,av_rt(3,:),'r--*',outlier_ratio,av_rt(4,:),'r--o',...
    outlier_ratio,av_rt(5,:),'b--+',outlier_ratio,av_rt(6,:),'b--^',outlier_ratio,av_rt(7,:),'b--*',outlier_ratio,av_rt(8,:),'b--o','linewidth',1); 
xlabel('Outlier ratio'); grid on; ylabel('Runtime (seconds)')
box_vars = findall(gca,'Type','Line');
legend(box_vars([8:-1:5]),'AM (Random init.)','AM (Huber init.)','AM (L1 init.)','AM (RANSAC init.)', 'Location', 'Best');
ah=axes('position',get(gca,'position'),'visible','off');
legend(ah,box_vars([4:-1:1]),'AM-R (Random init.)','AM-R (Huber init.)','AM-R (L1 init.)','AM-R (RANSAC init.)', 'Location', 'Best');
