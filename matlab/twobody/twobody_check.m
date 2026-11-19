function R = twobody_check()
%TWOBODY_CHECK Compare the linear burn-to-miss map with two-body propagation and trace the effect.
%   R = twobody_check() returns the 3x3 tables (indexed a1, a2) of the propagated miss vector,
%   the collision probabilities, the incentive and coordinator-rule consequences and the Monte
%   Carlo cross-check, together with propagator diagnostics and the settings used.
root = setup_paths(); %#ok<NASGU>
P = load_params();
E = load_expected();
S = load_scenario();
tb = P.two_body_check;
u = S.burns(:);
labels = string(P.action_labels);

names = {'xi_linear', 'xi_twobody', 'zeta_twobody', 'xi_lin_axes', 'zeta_lin_axes', 'tca_lin', ...
    'tca', 'along', 'xi_kepler', 'zeta_kepler', 'xi_second_order', ...
    'mc_pc', 'mc_se', 'mc_z', 'mc_hit_fraction', 'mc_plane_agreement', 'mc_pc_foster'};
for i = 1:numel(names)
    R.(names{i}) = zeros(3, 3);
end
outs = cell(3, 3);

fprintf('step 1: propagating the 9 joint actions (ode113 and Kepler)\n');
for a1 = 1:3
    for a2 = 1:3
        o = two_body_miss(S, u(a1), u(a2), 'ode');
        k = two_body_miss(S, u(a1), u(a2), 'kepler');
        outs{a1, a2} = o;
        R.xi_linear(a1, a2) = miss_shift(S, u(a1), u(a2));
        R.xi_twobody(a1, a2) = o.xi;
        R.zeta_twobody(a1, a2) = o.zeta;
        R.xi_lin_axes(a1, a2) = o.xi_lin_axes;
        R.zeta_lin_axes(a1, a2) = o.zeta_lin_axes;
        R.tca_lin(a1, a2) = o.tca_lin;
        R.tca(a1, a2) = o.tca;
        R.along(a1, a2) = o.along;
        R.xi_kepler(a1, a2) = k.xi;
        R.zeta_kepler(a1, a2) = k.zeta;
        R.xi_second_order(a1, a2) = second_order_xi(S, u(a1), u(a2));
    end
end
[U1, U2] = ndgrid(u, u);
R.d_xi = R.xi_twobody - R.xi_linear;
R.d_zeta = R.zeta_twobody - S.miss_zeta;
R.resid_second_order = R.xi_twobody - R.xi_second_order;
den = S.gain * (U1 + U2);
rel = R.d_xi ./ den;
rel(den == 0) = NaN;
R.rel_shift_error = rel;

fprintf('step 2: collision probability and incentives\n');
T_lin = cost_matrices(S);
R.pc_linear = T_lin.pc;
R.pc_twobody = pc2d(R.xi_twobody, R.zeta_twobody, S.sigma_xi, S.sigma_zeta, S.hbr);
R.rel_pc_change = R.pc_twobody ./ R.pc_linear - 1;
T_tb = cost_matrices(S, R.pc_twobody);

R.incentives = struct();
[b1l, b2l] = best_responses(T_lin);
[b1t, b2t] = best_responses(T_tb);
R.incentives.best_response_lin = struct('op1_given_a2', b1l, 'op2_given_a1', b2l);
R.incentives.best_response_twobody = struct('op1_given_a2', b1t, 'op2_given_a1', b2t);
R.incentives.n_best_response_changes = sum(b1l ~= b1t) + sum(b2l ~= b2t);
gaps = zeros(6, 1);
for j = 1:3
    c = sort(T_lin.c1(:, j));
    gaps(j) = c(2) - c(1);
    c = sort(T_lin.c2(j, :));
    gaps(3 + j) = c(2) - c(1);
end
R.incentives.smallest_incentive_gap = min(gaps);
R.incentives.largest_pc_change = max(abs(R.pc_twobody(:) - R.pc_linear(:)));
nl = pure_nash(T_lin);
nt = pure_nash(T_tb);
R.incentives.pure_nash_linear = nl;
R.incentives.pure_nash_twobody = nt;
R.incentives.pure_nash_equal = isequal(nl, nt);
[~, ~, pdl] = mixed_nash(T_lin.c1, T_lin.c2);
[~, ~, pdt] = mixed_nash(T_tb.c1, T_tb.c2);
R.incentives.p_dual_linear = pdl;
R.incentives.p_dual_twobody = pdt;

fprintf('step 3: recommendation rules of the baseline against the propagated probabilities\n');
[A_lin, Pm] = obedience_system(T_lin.c1, T_lin.c2);
A_tb = obedience_system(T_tb.c1, T_tb.c2);
scheme_names = {'public', 'private_m0', 'private_uniform', 'private_targeted_3', 'private_targeted_6'};
R.schemes = struct();
for i = 1:numel(scheme_names)
    nm = scheme_names{i};
    sc = E.baseline.schemes.(nm);
    mu = sc.margin;
    if strcmp(sc.margin_rows, 'targeted')
        mask = logical(E.baseline.obedience.targeted_mask(:));
    else
        mask = true(12, 1);
    end
    xf = reshape(sc.table.', [], 1);
    R.schemes.(nm) = struct('margin', mu, 'margin_rows', sc.margin_rows, ...
        'g_linear', max((A_lin + mu * (mask .* Pm)) * xf), ...
        'g_twobody', max((A_tb + mu * (mask .* Pm)) * xf), ...
        'h_twobody', max(A_tb * xf));
end

fprintf('step 4: re-solved correlated equilibrium\n');
[x_lin, f_lin] = solve_ce(T_lin.c1, T_lin.c2, T_lin.obj);
[x_tb, f_tb] = solve_ce(T_tb.c1, T_tb.c2, T_tb.obj);
R.resolve = struct('f_lin', f_lin, 'f_tb', f_tb, 'f_diff', f_tb - f_lin, ...
    'max_abs_dx', max(abs(x_tb(:) - x_lin(:))), 'x_lin', x_lin, 'x_tb', x_tb);

fprintf('step 5: importance-sampled Monte Carlo, 9 cells\n');
for a1 = 1:3
    for a2 = 1:3
        mc = pc_montecarlo(S, outs{a1, a2}, tb.mc_samples, tb.mc_seed + 3 * (a1 - 1) + (a2 - 1));
        R.mc_pc(a1, a2) = mc.pc;
        R.mc_se(a1, a2) = mc.se;
        R.mc_z(a1, a2) = mc.z;
        R.mc_hit_fraction(a1, a2) = mc.hit_fraction;
        R.mc_plane_agreement(a1, a2) = mc.plane_agreement;
        R.mc_pc_foster(a1, a2) = mc.pc_foster;
    end
end
R.naive = pc_montecarlo_naive(S, outs{1, 1}, tb.mc_samples, tb.mc_seed + 100);

fprintf('step 6: propagator diagnostics\n');
[r1, v1, r2, v2, B] = encounter_states(S);
R.nominal_axes = B;
[ro1, ~] = propagate_ode(r1, v1, -S.T_lead, S.mu, tb.ode_reltol, tb.ode_abstol);
[ro2, ~] = propagate_ode(r2, v2, -S.T_lead, S.mu, tb.ode_reltol, tb.ode_abstol);
[rk1, vk1] = propagate_kepler(r1, v1, -S.T_lead, S.mu);
[rk2, vk2] = propagate_kepler(r2, v2, -S.T_lead, S.mu);
R.prop_diff_m = [norm(ro1 - rk1), norm(ro2 - rk2)];
[rb1, ~] = propagate_kepler(rk1, vk1, S.T_lead, S.mu);
[rb2, ~] = propagate_kepler(rk2, vk2, S.T_lead, S.mu);
R.kepler_roundtrip_m = [norm(rb1 - r1), norm(rb2 - r2)];

R.settings = tb;
R.settings.matlab_version = version;
end

function [b1, b2] = best_responses(T)
% Best own action of operator 1 for each action of operator 2 (b1) and vice versa (b2).
b1 = zeros(3, 1);
b2 = zeros(3, 1);
for j = 1:3
    [~, b1(j)] = min(T.c1(:, j));
    [~, b2(j)] = min(T.c2(j, :));
end
end

function cells = pure_nash(T)
% Joint actions [a1 a2] (one-based) where neither operator can lower its own cost alone.
cells = zeros(0, 2);
for a1 = 1:3
    for a2 = 1:3
        if T.c1(a1, a2) <= min(T.c1(:, a2)) && T.c2(a1, a2) <= min(T.c2(a1, :))
            cells(end + 1, :) = [a1 a2]; %#ok<AGROW>
        end
    end
end
end
