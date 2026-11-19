function ca = closest_approach(r1, v1, r2, v2, mu, propagator, tol)
%CLOSEST_APPROACH True time of closest approach of two two-body orbits by Newton iteration.
%   Solves f(t) = dot(rho(t), w(t)) = 0 starting at the straight-line time, always propagating
%   from the given states (never chaining); step = -f / (w.w + rho.arel). propagator is 'ode'
%   or 'kepler'. Returns t, t_lin, states and rho, w at the converged time.
if strcmp(propagator, 'ode')
    prop = @(r, v, t) propagate_ode(r, v, t, mu);
else
    prop = @(r, v, t) propagate_kepler(r, v, t, mu);
end
rho0 = r2 - r1;
w0 = v2 - v1;
t_lin = -dot(rho0, w0) / dot(w0, w0);
t = t_lin;
for it = 1:20
    [a, b] = prop(r1, v1, t);
    [c, d] = prop(r2, v2, t);
    rho = c - a;
    w = d - b;
    f = dot(rho, w);
    arel = -mu * (c / norm(c)^3 - a / norm(a)^3);
    step = -f / (dot(w, w) + dot(rho, arel));
    t = t + step;
    if abs(step) < tol
        break
    end
end
[a, b] = prop(r1, v1, t);
[c, d] = prop(r2, v2, t);
ca = struct('t', t, 't_lin', t_lin, 'r1', a, 'v1', b, 'r2', c, 'v2', d, 'rho', c - a, 'w', d - b);
end
