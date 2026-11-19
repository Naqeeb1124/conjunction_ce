function T = cost_matrices(S, pc)
%COST_MATRICES Payoff tables over the 3x3 joint burn actions.
%   T = cost_matrices(S) or T = cost_matrices(S, pc). Rows index operator 1's action and
%   columns operator 2's action (no burn, small, large). pc defaults to the collision
%   probability at the shifted miss. Operator costs are fractions of satellite value;
%   obj is the coordinator cost in m/s.
u = S.burns(:);
[U1, U2] = ndgrid(u, u);
[xi, ~] = miss_shift(S, U1, U2);
if nargin < 2 || isempty(pc)
    pc = pc2d(xi, S.miss_zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
end
% Operator i fraction: (fuel cost of its burn + disruption if it burns) / satellite value
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
