function [x, fval, exitflag] = solve_ce(c1, c2, obj, mu, mask)
if nargin < 4 || isempty(mu)
    mu = 0;
end
if nargin < 5 || isempty(mask)
    mask = true(12, 1);
end
[A, Pm] = obedience_system(c1, c2);
M = A + mu * (mask(:) .* Pm);
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
