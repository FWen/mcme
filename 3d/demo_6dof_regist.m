clear all; close all; clc;

addpath data common ../minFunc ../minFunc/compiled

load('data_stanford_1')

% normalizing to [0, 1]^3 and downsampling
pc_o = double(conf.v1.Location);
pc_o = pc_o - repmat(min(pc_o),size(pc_o,1),1);
pc_o = pointCloud(pc_o./repmat(max(pc_o),size(pc_o,1),1));

N = 200;
sigma = 0.1;
outlier_ratios = [0:0.2:0.6, 0.7, 0.8, 0.9, 0.92, 0.95];

for n=1:length(outlier_ratios)
    for k=1:50
        
        [outlier_ratios(n), k]
        
        J = randperm(pc_o.Count);
        kpc1 = pointCloud(pc_o.Location(J(1:N),:));
        
        rng('shuffle')
        G = affine3d([randrot, zeros(3,1); 200*rand(1,3)-100, 1]); % rotation + translation
        kpc2 = pctransform(kpc1,G);
        
        
        %% noise and outlier generation
        n_outl = round(N*outlier_ratios(n));
        pt1 = double(kpc1.Location);
        pt2 = double(kpc2.Location) + randn(size(pt1))*sigma;
        J = randperm(N);
        pt2(J(1:n_outl),:) = rand(n_outl,3)*5.0;
        

        %% RANSAC
        [nInliers, T, iter, ~] = ransac6dof_timeout(pt1, pt2, 3*sigma, .99);
        T = T';
        angErr1(k,n) = 2*asin(norm(G.T(1:3,1:3)-T(1:3,1:3),'fro')/(2*sqrt(2)));
        trErr1(k,n)  = norm(G.T(1:3,1:3)*G.T(4,1:3)'-T(1:3,1:3)*T(4,1:3)');
        q_ransac = dcm2quat(T(1:3,1:3));
        t_ransac = T(4,1:3)';

        
        %%  MCME
        [R, t] = mcme_6dof_regist(pt1, pt2, sigma, q_ransac, t_ransac);
        angErr2(k,n) = 2*asin(norm(G.T(1:3,1:3)-R,'fro')/(2*sqrt(2)));
        trErr2(k,n)  = norm(G.T(1:3,1:3)*G.T(4,1:3)'-R*t);
    end
end



%% rotation error
m = 2;
x = [angErr1; angErr2]/pi*180;
g1 = kron([1:m]', ones(size(angErr1))); g1 = g1(:);
g2 = repmat(1:n,size(x,1),1); g2 = g2(:);

f=figure(1);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','bm');
xlabel('Outlier ratio');
ylabel('Rotation error (deg)');
set(gca,'yscale','log');grid on;%ylim([1e-3,3e0]);
set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(angErr1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(angErr1,2)+2), 0.075, 0.035, 0.075],...
    'String',{outlier_ratios(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '})
box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'RANSAC','MCME'}, 'Location', 'Best');


%% translation error
x = [trErr1; trErr2];
g1 = kron([1:m]', ones(size(angErr1))); g1 = g1(:);
g2 = repmat(1:n,size(x,1),1); g2 = g2(:);

f=figure(2);
bh=boxplot(x(:), {g2,g1},'whisker',1,'colorgroup',g1, 'factorgap',[m, 1],'symbol','.','outliersize',4,'widths',0.6,...
    'factorseparator',[1],'colors','bm');
xlabel('Outlier ratio');
ylabel('Translation error');
set(gca,'yscale','log');grid on;%ylim([1e-3,3e0]);
set(bh,'linewidth',1.5);
set(findobj(gca,'tag','Outliers'),'Marker','+');  

for k=1:size(angErr1,2)
    annotation(f,'textbox',[((k-1)+1.5)/(size(angErr1,2)+2), 0.075, 0.035, 0.075],...
    'String',{outlier_ratios(k)},'FitBoxToText','off', 'EdgeColor','none');
end
set(gca,'XTickLabel',{' '})
box_vars = findall(gca,'Tag','Box');
hLegend = legend(box_vars([m:-1:1]), {'RANSAC','MCME'}, 'Location', 'Best');
