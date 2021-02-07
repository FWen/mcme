clear all; close all; clc;

addpath data common ../minFunc ../minFunc/compiled

dataset = 'stanford';
modelIdx = 1;

load(sprintf('data_%s_%d', dataset, modelIdx))

% normalizing to [0, 1]^3 and downsampling
pc_o = double(conf.v1.Location);
pc_o = pc_o - repmat(min(pc_o),size(pc_o,1),1);
pc_o = pointCloud(pc_o./repmat(max(pc_o),size(pc_o,1),1));

N = 100;
sigma = 0.01;
outlier_ratios = [0:0.2:0.6, 0.7, 0.8, 0.9, 0.92, 0.95];

for n=1:length(outlier_ratios)
    for k=1:50
        
        [outlier_ratios(n), k]
        
        J = randperm(pc_o.Count);
        kpc1 = pointCloud(pc_o.Location(J(1:N),:));

        rng('shuffle')
        G = affine3d([randrot, zeros(3,1); zeros(1,3), 1]); % rotation
        kpc2 = pctransform(kpc1,G);
        
        
        %% noise and outlier generation
        n_outl = round(N*outlier_ratios(n));
        pt1 = double(kpc1.Location);
        pt2 = double(kpc2.Location) + reshape(getGGD(3*N,0.5),N,3)*sigma; % Generalized Gaussian noise
        J = randperm(N);
        pt2(J(1:n_outl),:) = pt2(J(1:n_outl),:) + rand(n_outl,3)*5.0;
        

        %% RANSAC
        RANSAC_TIMEOUT = 60;
        t0 = tic;
        [nInliers, T, iter, ~] = ransac3dof_timeout(pt1, pt2, 2*sigma, .999, RANSAC_TIMEOUT);
        runtime1(k,n) = toc(t0);
        T = T'; 
        angErr1(k,n) = 2*asin(norm(G.T(1:3,1:3)-T(1:3,1:3),'fro')/(2*sqrt(2))); 
        q_ransac = dcm2quat(T(1:3,1:3)); 
       
        
        %% AM - LS loss
        tic
        [R] = sime_rot_regist(pt1, pt2, sigma, q_ransac);
        runtime2(k,n) = toc();
        angErr2(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));
        
        %%  AM - Lp loss, p=1.0
        tic
        [R] = sime_rot_regist_lp(pt1, pt2, sigma, 1.0, q_ransac);
        runtime3(k,n) = toc();
        angErr3(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));
        
        %%  AM - Lp loss, p=1.2
        tic
        [R] = sime_rot_regist_lp(pt1, pt2, sigma, 1.2, q_ransac);
        runtime4(k,n) = toc();
        angErr4(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));
        
        %%  AM - Lp loss, p=1.5
        tic
        [R] = sime_rot_regist_lp(pt1, pt2, sigma, 1.5, q_ransac);
        runtime5(k,n) = toc();
        angErr5(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));
        
        %%  AM - Lp loss, p=1.8
        tic
        [R] = sime_rot_regist_lp(pt1, pt2, sigma, 1.8, q_ransac);
        runtime6(k,n) = toc();
        angErr6(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));
    end
end

%% rotation error
m=5;
x = [angErr2; angErr3; angErr4; angErr5; angErr6]/pi*180;
g1 = kron([1:m]', ones(size(angErr2))); g1 = g1(:);
g2 = repmat(1:n,size(x,1),1); g2 = g2(:);

f=figure(1);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[5, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgkm');
xlabel('Outlier ratio'); ylabel('Rotation error (deg)');
set(gca,'yscale','log');grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(angErr2,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(angErr1,2)+2), 0.095, 0.035, 0.075],...
    'String',{outlier_ratios(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'LS','Lp (p=1)','Lp (p=1.2)','Lp (p=1.5)','Lp (p=1.8)'}, 'Location', 'Best');
