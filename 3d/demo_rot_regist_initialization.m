clear all; close all; clc;

addpath data common ../minFunc ../minFunc/compiled
load('data_stanford_1')

% normalizing to [0, 1]^3 and downsampling
pc_o = double(conf.v1.Location);
pc_o = pc_o - repmat(min(pc_o),size(pc_o,1),1);
pc_o = pointCloud(pc_o./repmat(max(pc_o),size(pc_o,1),1));

N = 100;
sigma = 0.01;
outlier_ratios = [0:0.2:0.6, 0.7, 0.8, 0.9, 0.92, 0.95];

for n=1:length(outlier_ratios)
    parfor k=1:50
        
        [outlier_ratios(n), k]
        
        J = randperm(pc_o.Count);
        kpc1 = pointCloud(pc_o.Location(J(1:N),:));

        rng('shuffle')
        G = affine3d([randrot, zeros(3,1); zeros(1,3), 1]); % rotation
        kpc2 = pctransform(kpc1,G);
        
        
        %% noise and outlier generation
        n_outl = round(N*outlier_ratios(n));
        pt1 = double(kpc1.Location);
        pt2 = double(kpc2.Location) + randn(size(pt1))*sigma;
        J = randperm(N);
        pt2(J(1:n_outl),:) = rand(n_outl,3)*5.0;
        
        
        %% RANSAC
        RANSAC_TIMEOUT = 60;
        t0 = tic;
        [nInliers, T, iter, ~] = ransac3dof_timeout(pt1, pt2, 2*sigma, .999, RANSAC_TIMEOUT);
        runtime_ransac = toc(t0);
        T = T'; 
        angErr_rs(k,n) = 2*asin(norm(G.T(1:3,1:3)-T(1:3,1:3),'fro')/(2*sqrt(2))); 
        q_ransac = dcm2quat(T(1:3,1:3)); 

        %% GORE
        rep_flag=1;
        t0 = tic;
        [H, T, ~] = gore6(pt1', pt2', 3*sigma, 0, rep_flag); 
        runtime_gore = toc(t0);
        T=T';
        angErr_gore(k,n) = 2*asin(norm(G.T(1:3,1:3)-T(1:3,1:3),'fro')/(2*sqrt(2)));  
        q_gore = dcm2quat(T(1:3,1:3)); 

      
        %% AM   - without init
        tic
        [R] = sime_rot_regist(pt1, pt2, sigma);
        runtime1(k,n) = toc();
        angErr1(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  

        %% AM   -  Huber init
        tic
        [R] = sime_rot_regist_huber_init(pt1, pt2, sigma);
        runtime2(k,n) = toc();
        angErr2(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  

        %% AM   -  GORE init
        tic
        [R] = sime_rot_regist(pt1, pt2, sigma, q_gore);
        runtime3(k,n) = toc() + runtime_gore;
        angErr3(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  
        
        %% AM   -  RANSAC init
        tic
        [R] = sime_rot_regist(pt1, pt2, sigma, q_ransac);
        runtime4(k,n) = toc() + runtime_ransac;
        angErr4(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  
        
        
        %% AM-R  - without init
        tic
        [R] = sime_rot_regist_sdr(pt1, pt2, sigma);
        runtime5(k,n) = toc();
        angErr5(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  

        %% AM-R  - Huber init
        tic
        [R] = sime_rot_regist_sdr_huber_init(pt1, pt2, sigma);
        runtime6(k,n) = toc();
        angErr6(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  
        
        %%  AM-R  - GORE init
        tic
        [R] = sime_rot_regist_sdr(pt1, pt2, sigma, q_gore);
        runtime7(k,n) = toc() + runtime_gore;
        angErr7(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  
        
        %%  AM-R  - RANSAC init
        tic
        [R] = sime_rot_regist_sdr(pt1, pt2, sigma, q_ransac);
        runtime8(k,n) = toc() + runtime_ransac;
        angErr8(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));  
    end
end

%% rotation error
m=2;
x = [angErr1; angErr5]/pi*180;
g1 = kron([1:m]', ones(size(angErr4))); g1 = g1(:);
g2 = repmat(1:n,size(x,1),1); g2 = g2(:);

f=figure(1);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Rotation error (deg)');
set(gca,'yscale','log');grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(angErr1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(angErr1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratios(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box'); ylim([0.015,1500]);
hLegend = legend(box_vars([m:-1:1]), {'AM (Random init.)','AM-R (Random init.)'}, 'Location', 'Best');

%%
x = [angErr2; angErr6]/pi*180;
g1 = kron([1:m]', ones(size(angErr4))); g1 = g1(:);
g2 = repmat(1:n,size(x,1),1); g2 = g2(:);

f=figure(2);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Rotation error (deg)');
set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(angErr1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(angErr1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratios(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '});box_vars = findall(gca,'Tag','Box');ylim([0.015,1500]);
hLegend = legend(box_vars([m:-1:1]), {'AM (Huber init.)','AM-R (Huber init.)'}, 'Location', 'Best');

%%
x = [angErr3; angErr7]/pi*180;
g1 = kron([1:m]', ones(size(angErr4))); g1 = g1(:);
g2 = repmat(1:n,size(x,1),1); g2 = g2(:);

f=figure(3);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Rotation error (deg)');
set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(angErr1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(angErr1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratios(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box'); ylim([0.015,1500]);
hLegend = legend(box_vars([m:-1:1]), {'AM (GORE init.)','AM-R (GORE init.)'}, 'Location', 'Best');

%%
x = [angErr4; angErr8]/pi*180;
g1 = kron([1:m]', ones(size(angErr4))); g1 = g1(:);
g2 = repmat(1:n,size(x,1),1); g2 = g2(:);

f=figure(4);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','rbgckm');
xlabel('Outlier ratio'); ylabel('Rotation error (deg)');
set(gca,'yscale','log'); grid on; set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(angErr1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(angErr1,2)+2.5), 0.1, 0.035, 0.075],...
    'String',{outlier_ratios(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '}); box_vars = findall(gca,'Tag','Box'); ylim([0.015,1500]);
hLegend = legend(box_vars([m:-1:1]), {'AM (RANSAC init.)','AM-R (RANSAC init.)'}, 'Location', 'Best');
