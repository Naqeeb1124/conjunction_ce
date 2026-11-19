function test_core()
%TEST_CORE Checks of the core port against the reference numbers in expected.json.
E = load_expected();
B = E.baseline;
S = load_scenario();
T = cost_matrices(S);

% a. derived quantities
check_close('derived orbit and gain', [S.a S.n S.T S.T_lead S.v S.gain], ...
    [B.derived.sma_m B.derived.mean_motion_rad_s B.derived.period_s B.derived.lead_time_s ...
     B.derived.circular_speed_mps B.derived.gain_m_per_mps], 1e-6);
check_close('fuel price per m/s', S.fuel_price, B.derived.fuel_price_usd_per_mps, 1e-6);

% b. miss table
check_close('miss_xi table', T.xi, B.miss_xi_table, 1e-6);

% c. collision probability table
check_close('pc table', T.pc, B.pc_table, 1e-3);

% d. maneuver fractions and costs
check_close('maneuver cost fraction', T.frac, B.maneuver_cost_fraction, 1e-6);
check_close('cost operator 1', T.c1, B.cost_op1, 1e-6);
check_close('cost operator 2', T.c2, B.cost_op2, 1e-6);
check_close('coordinator cost table', T.obj, B.coordinator_cost_table, 1e-6);

% e. obedience system
[A, Pm, info] = obedience_system(T.c1, T.c2);
check_close('obedience A', A, B.obedience.A, 1e-6);
check_close('obedience P', Pm, B.obedience.P, 1e-6);
check_close('obedience info', info, B.obedience.info + 1, 0);

% f. correlated equilibrium under each margin scheme
names = {'private_m0', 'private_uniform', 'private_targeted_3', 'private_targeted_6'};
for k = 1:numel(names)
    name = names{k};
    sch = B.schemes.(name);
    mu = sch.margin;
    if strcmp(sch.margin_rows, 'targeted')
        mask = logical(B.obedience.targeted_mask);
        mask = mask(:);
    else
        mask = true(12, 1);
    end
    [x, fval] = solve_ce(T.c1, T.c2, T.obj, mu, mask);
    check_close([name ' CE table'], x, sch.table, 0, 1e-6);
    check_close([name ' CE objective'], fval, sch.coordinator_cost, 1e-6);
    if strcmp(name, 'private_m0')
        check_true([name ' obedience satisfied'], ...
            max(A * reshape(x.', [], 1)) <= 1e-9, ...
            sprintf('max constraint value %.3g', max(A * reshape(x.', [], 1))));
    end
end

% g. mixed Nash equilibrium
[p, q, p_dual] = mixed_nash(T.c1, T.c2);
check_close('mixed Nash p', p, B.mixed_nash.p, 1e-6);
check_close('mixed Nash q', q, B.mixed_nash.q, 1e-6);
check_close('mixed Nash p_dual', p_dual, B.mixed_nash.p_dual_inaction, 1e-6);
costL1 = T.c1(3, :) * q;
costN1 = T.c1(1, :) * q;
costL2 = T.c2(:, 3).' * p;
costN2 = T.c2(:, 1).' * p;
check_true('large burn not profitable for operator 1 or 2', ...
    costL1 >= costN1 && costL2 >= costN2, ...
    sprintf('op1 L-N %.3g, op2 L-N %.3g', costL1 - costN1, costL2 - costN2));

% h. pc2d against direct 2D integration over the disc
R = S.hbr;
cells = [1 1; 2 1];
for c = 1:size(cells, 1)
    xm = T.xi(cells(c, 1), cells(c, 2));
    ym = S.miss_zeta;
    sx = S.sigma_xi;
    sy = S.sigma_zeta;
    dens = @(x, y) exp(-0.5 * ((x - xm) / sx).^2 - 0.5 * ((y - ym) / sy).^2) / (2 * pi * sx * sy);
    ref = integral2(@(r, th) dens(r .* cos(th), r .* sin(th)) .* r, 0, R, 0, 2 * pi, ...
        'RelTol', 1e-10, 'AbsTol', 0);
    got = pc2d(xm, ym, sx, sy, R);
    check_close(sprintf('pc2d vs integral2 cell (%d,%d)', cells(c, 1), cells(c, 2)), got, ref, 1e-6);
end

% i. symmetry in the axes and array broadcasting
xm0 = S.miss_xi; ym0 = S.miss_zeta;
check_close('pc2d axis swap symmetry', ...
    pc2d(xm0, ym0, S.sigma_xi, S.sigma_zeta, R), ...
    pc2d(ym0, xm0, S.sigma_zeta, S.sigma_xi, R), 1e-12);
Xarr = T.xi;
Parr = pc2d(Xarr, S.miss_zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
Pscalar = zeros(3, 3);
for a1 = 1:3
    for a2 = 1:3
        Pscalar(a1, a2) = pc2d(Xarr(a1, a2), S.miss_zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
    end
end
check_close('pc2d array equals scalar calls', Parr, Pscalar, 1e-12);

% j. gallery cases
P = load_params();
PCg = zeros(3, 3, 12);
Gpc = zeros(3, 3, 12);
fg = zeros(12, 1);
Gobj = zeros(12, 1);
for i = 1:12
    Si = load_scenario(P.gallery(i).scenario, P.constants);
    Ti = cost_matrices(Si);
    [~, fi] = solve_ce(Ti.c1, Ti.c2, Ti.obj);
    PCg(:, :, i) = Ti.pc;
    fg(i) = fi;
    Gpc(:, :, i) = E.gallery(i).pc_table;
    Gobj(i) = E.gallery(i).ce_objective;
end
check_close('gallery pc tables', PCg, Gpc, 1e-3);
check_close('gallery CE objectives', fg, Gobj, 1e-6);
end
