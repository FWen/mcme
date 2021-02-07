clear all; clc;

addpath src ../minFunc

N = 250;   % Number of points
d = 8;     % data dimension
sig = 0.1; % inlier std used to generate Gaussian noise
ons = 4;  % outlier noise strength

% Inlier threshold
inlier_th = sqrt(chi2inv(1-1e-3,1))*sig;

out_ratio = 0:0.1:0.8;
for nr=1:length(out_ratio)
    parfor k=1:100
        [out_ratio(nr), k]

        %% generate data
        A  = 2*rand(N,d) - 1;
        xt = rand(d,1);
        y  = A*xt + sig*getGGD(N,0.5); % generalized Gaussian noise
        n_outl = round(out_ratio(nr)*N);
        J = randperm(N);
        
        y(J(1:n_outl),:) = y(J(1:n_outl),:) + (2*rand(n_outl,1)-1)*ons; % uniformly distributed outliers
%        y(J(1:n_outl),:) = y(J(1:n_outl),:) + randn(n_outl,1)*ons/2; % Gaussian distributed outliers
        
        
        %% L1 method
        tic; 
        [l1Inliers, l1Theta] = l1_alg(A,y,inlier_th,1);
        Ai = A(l1Inliers,:);
        l1Theta_ref = inv(Ai.'*Ai)*Ai.'*y(l1Inliers);
        mc1(k) = numel(l1Inliers);
        Err1(k,nr) = norm(l1Theta-xt)/norm(xt);
        RefErr1(k,nr) = norm(l1Theta_ref-xt)/norm(xt);
        runtime1(k) = toc;
        
                
        %% AM - LS loss
        tic
        [mmTheta, mmInliers] = sime_linear(A, y, sig, l1Theta_ref);
        mc2(k) = numel(mmInliers);
        Err2(k,nr) = norm(mmTheta-xt)/norm(xt);
        runtime2(k) = runtime1(k)+toc;
        
        %% AM - Lp loss, p=1.0
        tic
        [mmTheta, mmInliers] = sime_linear_lp(A, y, sig, 1.0, l1Theta_ref);
        mc3(k) = numel(mmInliers);
        Err3(k,nr) = norm(mmTheta-xt)/norm(xt);
        runtime3(k) = runtime1(k)+toc;

        %% AM - Lp loss, p=1.2
        tic
        [mmTheta, mmInliers] = sime_linear_lp(A, y, sig, 1.2, l1Theta_ref);
        mc4(k) = numel(mmInliers);
        Err4(k,nr) = norm(mmTheta-xt)/norm(xt);
        runtime4(k) = runtime1(k)+toc;
        
        %% AM - Lp loss, p=1.5
        tic
        [mmTheta, mmInliers] = sime_linear_lp(A, y, sig, 1.5, l1Theta_ref);
        mc5(k) = numel(mmInliers);
        Err5(k,nr) = norm(mmTheta-xt)/norm(xt);
        runtime5(k) = runtime1(k)+toc;
        
        %% AM - Lp loss, p=1.8
        tic
        [mmTheta, mmInliers] = sime_linear_lp(A, y, sig, 1.8, l1Theta_ref);
        mc6(k) = numel(mmInliers);
        Err6(k,nr) = norm(mmTheta-xt)/norm(xt);
        runtime6(k) = runtime1(k)+toc;
    end
    av_mc(:,nr) = mean([mc1; mc2; mc3; mc4; mc5; mc6], 2);
    av_rt(:,nr) = mean([runtime1; runtime2; runtime3; runtime4; runtime5; runtime6], 2);
end

f=figure(1);
m = 5; x = [Err2; Err3; Err4; Err5; Err6];
g1 = kron([1:m]', ones(size(Err2,1),nr)); g1 = g1(:);
g2 = repmat(1:nr,size(x,1),1); g2 = g2(:);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[6, 1],'symbol','.','outliersize',4,'widths',0.9,...
    'factorseparator',[1],'colors','rbgkm'); 
xlabel('Outlier ratio'); ylabel('Model estimation error'); set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5); % ylim([1e-3,3e0]);
set(findobj(gca,'tag','Outliers'),'Marker','+');  
for k=1:nr
    annotation(f,'textbox',[((k-1)+1.5)/(nr+2), 0.11, 0.035, 0.075],...
    'String',{out_ratio(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'LS','Lp (p=1)','Lp (p=1.2)','Lp (p=1.5)','Lp (p=1.8)'}, 'Location', 'Best');

% figure(2);
% subplot(3,1,1); plot(getGGD(1e3,2)); title('GGD (v=2), Gaussian'); ylim([-5,5]);
% subplot(3,1,2); plot(getGGD(1e3,1)); title('GGD (v=1), Laplacian'); ylim([-5,5]);
% subplot(3,1,3); plot(getGGD(1e3,0.5)); title('GGD (v=0.5)'); ylim([-5,5]);
