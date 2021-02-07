function [f,g] = objfun(rr,par)

n = par.n;
p = par.p;
Q = par.Q;

R = reshape(rr,p, n).';

r1_ri = R(2:end,:)*R(1,:).';
r_norm = sum(R.^2,2).^0.5;


f = ( 2*Q(1,2:end) ./ (r_norm(1)*r_norm(2:end)).' ) * r1_ri;


tt1 = 2*Q(1,2:end) ./ (r_norm(1)^3*r_norm(2:end)).';
g1 = tt1 * (r_norm(1)^2*R(2:end,:) - r1_ri*R(1,:));

tt2 = 2*Q(1,2:end)' ./ (r_norm(1)*r_norm(2:end).^3);
g2 = ((r_norm(2:end).^2*R(1,:) - R(2:end,:).*repmat(r1_ri,1,p)) .* repmat(tt2,1,p)).';

g = [g1.'; g2(:)];
   
end
