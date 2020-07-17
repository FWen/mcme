clear all; close all;  clc;

addpath src ../minFunc;

inlier_th = 4 ; % Inlier threshold in pixel
    
% parameters for the EP method
solver =  'gurobi'; opt.lpsolver = prepareSolver(solver);    opt.solver = solver;
opt.QThresh = 1e-5;    opt.alpha = 10;   opt.kappa = 1.5;    opt.alphaMax = 1e9;
    
imgs={'AerialviewsI','Corridor','kapel', 'MertonCollegeII','MertonCollegeIII','ValbonneChurch',...
      'boat','bark','bikes','graf','trees',...
      'building4','building5','building22','building24','building28',...
      'building37','building59','building67','building199'};
   
for k=1:length(imgs)
    load(['dataset/', imgs{k}, '.mat']);
    Ns(k) = size(data.x1,2);

    th = inlier_th*data.T2(1,1); 
    [A0, b0, c0, d0] = genMatrixHomography(data.x1, data.x2);
    [A, b] = genLinearMatrixFromQuasiconvex(A0, b0, c0, d0, th);

    
    %% RANSAC
    [rsTheta, rsInliers, rsRuntime] = homographyFit(data, th, 'RANSAC', rand(8,1));
    mc(k,1) = length(rsInliers);
    rt(k,1) = rsRuntime;
    H21 = inv(data.T2)*reshape(rsTheta, [3,3])'*data.T1; % estimated homography
    scores(k,1) = compute_homo_score(H21, data.matches.X1, data.matches.X2);
    [H21,rsTheta_ref] = homo_refine(A,b,rsInliers,data);% refinement
    scores(k,2) = compute_homo_score(H21, data.matches.X1, data.matches.X2);


    %% L1 method
    tic; 
    [L1Inliers, L1Theta] = homo_L1(A,b);
    mc(k,2) = length(L1Inliers);
    rt(k,2) = toc;
    H21 = inv(data.T2)*reshape([L1Theta; 1], [3,3])'*data.T1; % estimated homography
    scores(k,3) = compute_homo_score(H21, data.matches.X1, data.matches.X2);
    [H21,L1Theta_ref] = homo_refine(A,b,L1Inliers,data);% refinement
    scores(k,4) = compute_homo_score(H21, data.matches.X1, data.matches.X2);


    %% Iteratively-Reweighted 
    tic; 
    [irwInliers,irwTheta] = homo_irw(A,b);
    mc(k,3) = length(irwInliers);
    rt(k,3) = toc;
    H21 = inv(data.T2)*reshape([irwTheta; 1], [3,3])'*data.T1; % estimated homography
    scores(k,5) = compute_homo_score(H21, data.matches.X1, data.matches.X2);
    H21 = homo_refine(A,b,irwInliers,data);% refinement
    scores(k,6) = compute_homo_score(H21, data.matches.X1, data.matches.X2);
    
    
    %% EP
    [eprsTheta, eprsInliers, eprsRuntime] = homographyFit(data, th, 'EP', rsTheta_ref, opt);
    mc(k,4) = length(eprsInliers);
    rt(k,4) = eprsRuntime+rsRuntime;
    H21 = inv(data.T2)*reshape([eprsTheta; 1], [3,3])'*data.T1; % estimated homography
    scores(k,7) = compute_homo_score(H21, data.matches.X1, data.matches.X2);
    H21 = homo_refine(A,b,eprsInliers,data);% refinement
    scores(k,8) = compute_homo_score(H21, data.matches.X1, data.matches.X2); 
    
        
    %% MCME
    tic
    [mmTheta, mmInliers] = mcme_nonlinear(A0, b0, c0, d0, [], rsTheta_ref, th);
    mc(k,5) = length(mmInliers);
    rt(k,5) = toc + rsRuntime;
    H21 = inv(data.T2)*reshape([mmTheta; 1], [3,3])'*data.T1; % estimated homography
    scores(k,9) = compute_homo_score(H21, data.matches.X1, data.matches.X2);
end

mc
mean(scores)

img_indx = 1:length(imgs);
figure(1);
p = semilogy(img_indx,scores(:,1),'r--+',img_indx,scores(:,2),'r--o',img_indx,scores(:,3),'g--+',...
    img_indx,scores(:,4),'g--o',img_indx,scores(:,5),'b--+',img_indx,scores(:,6),'b--o',...
    img_indx,scores(:,7),'c--+',img_indx,scores(:,8),'c--o',img_indx,scores(:,9),'m--^'); 
xlabel('Image index'); grid on;
ylabel('Score of estimated homography')
legend('RANSAC','RANSAC-r','L1','L1-r','IRW','IRW-r','EP','EP-r','MCME', 'Location', 'Best');

figure(2);
semilogy(img_indx,rt(:,1),'r--+',img_indx,rt(:,2),'g--*',img_indx,rt(:,3),'b--x',...
     img_indx,rt(:,4),'k--o',img_indx,rt(:,5),'m--^'); 
xlabel('Image index'); grid on;
ylabel('Runtime (second)')
legend('RANSAC','L1','IRW','EP','MCME','Location','Best');

figure(3);
semilogy(img_indx,mc(:,1),'r--+',img_indx,mc(:,2),'g--*',img_indx,mc(:,3),'b--x',...
         img_indx,mc(:,4),'k--o',img_indx,mc(:,5),'m--^'); 
xlabel('Image index'); grid on;
ylabel('Consensus size')
legend('RANSAC','L1','IRW','EP','MCME','Location','Best');
