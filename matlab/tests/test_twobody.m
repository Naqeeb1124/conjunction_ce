function test_twobody()
%TEST_TWOBODY Checks of the two-body propagation of the burn-to-miss map against the references.
root = setup_paths();
E = load_expected();
S = load_scenario();
R = twobody_check();
tp = E.baseline.two_body_python;

check_close('nominal xi axis', R.nominal_axes.xi, [0 1 0], 0, 1e-12);
check_close('nominal zeta axis', R.nominal_axes.zeta, [1 0 0], 0, 1e-12);

check_true('ode vs Kepler after lead time <= 1e-3 m', all(R.prop_diff_m <= 1e-3), ...
    sprintf('%.3e %.3e m', R.prop_diff_m));
check_true('Kepler round trip <= 1e-6 m', all(R.kepler_roundtrip_m <= 1e-6), ...
    sprintf('%.3e %.3e m', R.kepler_roundtrip_m));

check_close('xi straight-line vs python', R.xi_lin_axes, tp.miss_xi, 0, 1e-3);
check_close('zeta straight-line vs python', R.zeta_lin_axes, tp.miss_zeta, 0, 1e-3);
check_close('tca shift vs python', R.tca_lin, tp.tca_shift_s, 0, 1e-6);

check_close('no-burn cell is the nominal miss', [R.xi_twobody(1, 1) R.zeta_twobody(1, 1)], ...
    [S.miss_xi S.miss_zeta], 0, 1e-3);

check_close('xi two-body vs second-order formula', R.xi_twobody, R.xi_second_order, 0, 2e-4);
check_close('zeta two-body vs nominal zeta', R.zeta_twobody, S.miss_zeta * ones(3, 3), 0, 2e-4);

check_close('xi true closest approach vs straight-line', R.xi_twobody, R.xi_lin_axes, 0, 1e-4);
check_close('zeta true closest approach vs straight-line', R.zeta_twobody, R.zeta_lin_axes, 0, 1e-4);

check_close('xi Kepler vs ode', R.xi_kepler, R.xi_twobody, 0, 1e-3);
check_close('zeta Kepler vs ode', R.zeta_kepler, R.zeta_twobody, 0, 1e-3);

rel = R.rel_shift_error;
rel = rel(~isnan(rel));
check_true('linear map relative shift error <= 1e-4 (8 burn cells)', numel(rel) == 8 && all(abs(rel) <= 1e-4), ...
    sprintf('largest %.3e', max(abs(rel))));

[zmax, idx] = max(abs(R.mc_z(:)));
[i1, i2] = ind2sub([3 3], idx);
check_true('Monte Carlo max |z| <= 4', zmax <= 4, sprintf('largest |z| %.3f in cell (%d,%d)', zmax, i1, i2));
check_true('Monte Carlo plane agreement >= 0.999', min(R.mc_plane_agreement(:)) >= 0.999, ...
    sprintf('min %.5f', min(R.mc_plane_agreement(:))));

% Incentives: best replies and pure Nash equilibria under the propagated probabilities.
inc = R.incentives;
n = inc.n_best_response_changes;
check_true('two-body probabilities change no best reply', ...
    n == 0 && isequal(inc.best_response_lin, inc.best_response_twobody), ...
    sprintf('%d of 6 best replies change', n));
check_true('two-body probabilities keep the pure Nash equilibria', ...
    inc.pure_nash_equal && isequal(inc.pure_nash_linear, inc.pure_nash_twobody), ...
    sprintf('%d equilibria', size(inc.pure_nash_linear, 1)));
check_true('largest change of a collision probability is below 1e-3 of the smallest incentive gap', ...
    inc.largest_pc_change <= 1e-3 * inc.smallest_incentive_gap, ...
    sprintf('largest change %.3e, smallest gap %.3e, ratio %.0f', inc.largest_pc_change, ...
    inc.smallest_incentive_gap, inc.smallest_incentive_gap / inc.largest_pc_change));

% Obedience: h is the largest plain constraint value A*x, g adds the margin on the margin rows.
kept = {'public', 'private_uniform', 'private_targeted_3', 'private_targeted_6'};
h = cellfun(@(nm) R.schemes.(nm).h_twobody, kept);
withMargin = kept(2:end);
gfrac = cellfun(@(nm) R.schemes.(nm).g_twobody / R.schemes.(nm).margin, withMargin);
check_true('public rule and rules with a margin keep every obedience constraint with the two-body probabilities', ...
    all(h <= 1e-12), sprintf('largest h %.2e; g/margin %.2e %.2e %.2e', max(h), gfrac));
h0 = R.schemes.private_m0.h_twobody;
check_true('rule without a margin: the exact tie tips, by less than 1e-8', 0 < h0 && h0 <= 1e-8, ...
    sprintf('largest constraint value %.3e', h0));

% Coordinator rule re-solved with the propagated probabilities.
check_true('coordinator rule re-solved with the two-body probabilities moves by less than 1e-5', ...
    abs(R.resolve.f_diff) <= 1e-5 && R.resolve.max_abs_dx <= 1e-5, ...
    sprintf('cost change %.3e m/s, largest entry change %.3e', R.resolve.f_diff, R.resolve.max_abs_dx));

check_true('results/twobody_check.json exists', isfile(fullfile(root, 'results', 'twobody_check.json')), ...
    'run run_twobody_check first');
end
