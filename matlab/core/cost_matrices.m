function T = cost_matrices(S, pc)
u = S.burns(:);
[U1, U2] = ndgrid(u, u);
[xi, ~] = miss_shift(S, U1, U2);
if nargin < 2 || isempty(pc)
    pc = pc2d(xi, S.miss_zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
end
frac = zeros(2, 3);
for i = 1:2
    frac(i, :) = (S.fuel_price(i) * u.' + S.disruption(i) * (u.' > 0)) / S.value(i);
end
c1 = frac(1, :).' + pc;
c2 = frac(2, :) + pc;
dv_total = U1 + U2;
obj = dv_total + S.lam * pc;

T = struct('u1', U1, 'u2', U2, 'dv_total', dv_total, 'xi', xi, 'pc', pc, ...
    'frac', frac, 'c1', c1, 'c2', c2, 'obj', obj);
end
