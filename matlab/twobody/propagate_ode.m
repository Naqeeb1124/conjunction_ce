function [r, v] = propagate_ode(r0, v0, t, mu, reltol, abstol)
if nargin < 5 || isempty(reltol) || nargin < 6 || isempty(abstol)
    P = load_params();
    if nargin < 5 || isempty(reltol)
        reltol = P.two_body_check.ode_reltol;
    end
    if nargin < 6 || isempty(abstol)
        abstol = P.two_body_check.ode_abstol;
    end
end
if t == 0
    r = r0;
    v = v0;
    return
end
opts = odeset('RelTol', reltol, 'AbsTol', abstol);
rhs = @(~, y) [y(4:6); -mu * y(1:3) / norm(y(1:3))^3];
[~, y] = ode113(rhs, [0 t], [r0(:); v0(:)], opts);
r = y(end, 1:3);
v = y(end, 4:6);
end
