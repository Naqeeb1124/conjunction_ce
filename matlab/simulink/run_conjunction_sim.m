function res = run_conjunction_sim(S, x_rule, u_draw)
%RUN_CONJUNCTION_SIM Run the Simulink conjunction model for one recommended joint action.
%   res = run_conjunction_sim(S, x_rule, u_draw): S from load_scenario (default baseline),
%   x_rule a 3x3 table over (a1, a2) (default: coordinator rule), u_draw in [0,1) (default 0.5).
%   All numbers reach the model as variables of a Simulink.SimulationInput. The model returns
%   the closest-approach vector m (in space) and the final velocities. The projection
%   xi = m.B0.xi, zeta = m.B0.zeta is done here on the nominal axes B0 of encounter_states(S)
%   (the axes at closest approach when nobody burns). xi_axes_final, zeta_axes_final are the
%   components of m on encounter_basis(v1_final, v2_final). Fields: a1, a2, u1, u2, xi, zeta,
%   miss, t_ca, t, range, n_steps, m, w_final, xi_axes_final, zeta_axes_final.
root = setup_paths();
if nargin < 1 || isempty(S)
    S = load_scenario();
end
if nargin < 2 || isempty(x_rule)
    T = cost_matrices(S);
    x_rule = solve_ce(T.c1, T.c2, T.obj);
end
if nargin < 3 || isempty(u_draw)
    u_draw = 0.5;
end
P = load_params();
mdl = 'conjunction_sim';
if ~exist(fullfile(root, 'simulink', [mdl '.slx']), 'file')
    build_conjunction_sim();
end
load_system(fullfile(root, 'simulink', mdl));

% Pre-burn states: the nominal closest-approach states propagated back by the lead time.
[r1, v1, r2, v2, B0] = encounter_states(S);
[r1_0, v1_0] = propagate_kepler(r1, v1, -S.T_lead, S.mu);
[r2_0, v2_0] = propagate_kepler(r2, v2, -S.T_lead, S.mu);

in = Simulink.SimulationInput(mdl);
in = in.setVariable('x_rule_flat', reshape(x_rule.', [], 1));
in = in.setVariable('u_draw', u_draw);
in = in.setVariable('burns', S.burns(:));
in = in.setVariable('burn_sign', S.burn_sign(:));
in = in.setVariable('mu_earth', S.mu);
in = in.setVariable('r1_0', r1_0(:));
in = in.setVariable('v1_0', v1_0(:));
in = in.setVariable('r2_0', r2_0(:));
in = in.setVariable('v2_0', v2_0(:));
in = in.setVariable('T_lead', S.T_lead);
in = in.setVariable('ode_reltol', P.two_body_check.ode_reltol);
in = in.setVariable('ode_abstol', P.two_body_check.ode_abstol);
out = sim(in);
if ~isempty(out.ErrorMessage)
    error('run_conjunction_sim:simFailed', '%s', out.ErrorMessage);
end

t = out.tout(:);
range = out.range_log(:);
if numel(range) ~= numel(t) || abs(t(end) - S.T_lead) > 1e-9 * S.T_lead
    error('run_conjunction_sim:logging', 'Logged histories do not end at the stop time.');
end
vf1 = out.v1_log(end, :);
vf2 = out.v2_log(end, :);
B = encounter_basis(vf1, vf2);
m = out.m_log(end, :);

res.a1 = round(out.a1_log(end));
res.a2 = round(out.a2_log(end));
res.u1 = out.u1_log(end);
res.u2 = out.u2_log(end);
res.xi = dot(m, B0.xi);
res.zeta = dot(m, B0.zeta);
res.xi_axes_final = dot(m, B.xi);
res.zeta_axes_final = dot(m, B.zeta);
res.miss = norm(m);
res.t_ca = out.tcorr_log(end);
res.t = t;
res.range = range;
res.n_steps = numel(t) - 1;
res.m = m;
res.w_final = vf2 - vf1;
end
