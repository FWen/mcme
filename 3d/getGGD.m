
function [X] = getGGD(N,p)
 
X1 = zeros(N,1);
cc = p;
c  = p/2;
for i = 1:N
    %cc = 2 is Gaussian
    X1(i) = gamrnd(2/cc,1)^(1/cc) * exp(j*2*pi*rand);
end
cNorm = (gamma(2/c)/(gamma(1/c))); %unit complex norm
X1 =X1*(1/sqrt(cNorm)); %Complex to unit variance

X = real(X1);
X = X/std(X);