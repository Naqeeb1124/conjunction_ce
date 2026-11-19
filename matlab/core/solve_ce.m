function [x, fval, exitflag] = solve_ce(c1, c2, obj, mu, mask)
%SOLVE_CE Coordinator's correlated equilibrium as a linear program over the 3x3 joint table.
%   [x, fval, exitflag] = solve_ce(c1, c2, obj, mu, mask) minimises the coordinator cost
%   subject to the obedience constraints (A + mu * (mask .* Pm)) x <= 0, sum(x) = 1, x >= 0.
%   mu defaults to 0 and mask to all twelve rows. Infeasible programs return x = [] and
%   fval = Inf.
if nargin < 4 || isempty(mu)
    mu = 0;
end
if nargin < 5 || isempty(mask)
    mask = true(12, 1);
end
[A, Pm] = obedience_system(c1, c2);
M = A + mu * (mask(:) .* Pm);
% Scale each inequality row by its largest absolute entry; all-zero rows are left as they are
rowmax = max(abs(M), [], 2);
nz = rowmax > 0;
M(nz, :) = M(nz, :) ./ rowmax(nz);
b = zeros(size(M, 1), 1);
Aeq = ones(1, 9);
beq = 1;
lb = zeros(9, 1);
f = reshape(obj.', [], 1);

opts = optimoptions('linprog', 'Display', 'none', ...
    'ConstraintTolerance', 1e-9, 'OptimalityTolerance', 1e-9);
[xf, ~, exitflag] = linprog(f, M, b, Aeq, beq, lb, [], opts);
if exitflag == -2 || isempty(xf)
    x = [];
    fval = Inf;
    return
end
xf(xf < 1e-12) = 0;
xf = xf / sum(xf);
x = reshape(xf, 3, 3).';
fval = f.' * xf;
end
