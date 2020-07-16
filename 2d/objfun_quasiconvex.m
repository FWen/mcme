function [f,g] = objfun_quasiconvex(theta,par)

d = length(theta);

A = par.A; b = par.b;
c = par.c; w = par.w;
Ax = A(1:2:end,:);
Ay = A(2:2:end,:);

Abx = (b(1,:)' + Ax*theta);
Aby = (b(2,:)' + Ay*theta);
cxd = theta'*c+w;

f = par.s' * ((Abx.^2 + Aby.^2)'./cxd.^2)';

g = 2*par.s' * ((Ax.*repmat(Abx,1,d) + Ay.*repmat(Aby,1,d))./repmat(cxd.^2,d,1)' - repmat((Abx.^2 + Aby.^2)'./cxd.^3,d,1)'.*c');
g = g';   

end
