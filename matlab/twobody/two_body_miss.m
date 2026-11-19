function out = two_body_miss(S, u1, u2, propagator)
%TWO_BODY_MISS Miss vector after two tangential burns, by two-body propagation.
%   out = two_body_miss(S, u1, u2, propagator) with propagator 'ode' (default) or 'kepler'.
%   Both satellites go back by T_lead, burn along their velocity, then return to t = 0.
%   *_lin_axes: straight-line closest approach in the nominal axes; xi, zeta, along: true
%   closest approach in the axes built from the velocities at that time.
if nargin < 4 || isempty(propagator)
    propagator = 'ode';
end
P = load_params();
tb = P.two_body_check;
if strcmp(propagator, 'ode')
    prop = @(r, v, t) propagate_ode(r, v, t, S.mu, tb.ode_reltol, tb.ode_abstol);
else
    prop = @(r, v, t) propagate_kepler(r, v, t, S.mu);
end
[r1, v1, r2, v2, B0] = encounter_states(S);
[r1b, v1b] = prop(r1, v1, -S.T_lead);
[r2b, v2b] = prop(r2, v2, -S.T_lead);
v1b = v1b + S.burn_sign(1) * u1 * v1b / norm(v1b);
v2b = v2b + S.burn_sign(2) * u2 * v2b / norm(v2b);
[r1, v1] = prop(r1b, v1b, S.T_lead);
[r2, v2] = prop(r2b, v2b, S.T_lead);
rho = r2 - r1;
w = v2 - v1;
t_lin = -dot(rho, w) / dot(w, w);
m = rho + w * t_lin;
out.xi_lin_axes = dot(m, B0.xi);
out.zeta_lin_axes = dot(m, B0.zeta);
out.tca_lin = t_lin;
ca = closest_approach(r1, v1, r2, v2, S.mu, propagator, tb.tca_tol_s);
B = encounter_basis(ca.v1, ca.v2);
out.xi = dot(ca.rho, B.xi);
out.zeta = dot(ca.rho, B.zeta);
out.along = dot(ca.rho, B.s);
out.miss = norm(ca.rho);
out.tca = ca.t;
out.r1 = ca.r1;
out.v1 = ca.v1;
out.r2 = ca.r2;
out.v2 = ca.v2;
out.B = B;
end
