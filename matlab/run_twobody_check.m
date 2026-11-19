root = setup_paths();
R = twobody_check();
fid = fopen(fullfile(root, 'results', 'twobody_check.json'), 'w');
fwrite(fid, jsonencode(R, 'PrettyPrint', true));
fclose(fid);

P = load_params();
labels = string(P.action_labels);
fprintf('cell  xi_lin    xi_2body  dxi_mm   dzeta_mm  pc_lin      pc_2body    rel_pc     mc_pc       mc_z\n');
for a1 = 1:3
    for a2 = 1:3
        fprintf('%s%s  %8.3f  %8.3f  %7.4f  %8.4f  %.4e  %.4e  %+.2e  %.4e  %+.2f\n', ...
            labels(a1), labels(a2), R.xi_linear(a1, a2), R.xi_twobody(a1, a2), ...
            1e3 * R.d_xi(a1, a2), 1e3 * R.d_zeta(a1, a2), R.pc_linear(a1, a2), ...
            R.pc_twobody(a1, a2), R.rel_pc_change(a1, a2), R.mc_pc(a1, a2), R.mc_z(a1, a2));
    end
end
I = R.incentives;
fprintf('best-response changes %d, smallest incentive gap %.3e, largest pc change %.3e, pure Nash equal %d\n', ...
    I.n_best_response_changes, I.smallest_incentive_gap, I.largest_pc_change, I.pure_nash_equal);
fprintf('p_dual linear %.6f, two-body %.6f\n', I.p_dual_linear, I.p_dual_twobody);
for nm = fieldnames(R.schemes).'
    s = R.schemes.(nm{1});
    fprintf('%-20s g_linear %+.3e  g_twobody %+.3e  h_twobody %+.3e\n', nm{1}, ...
        s.g_linear, s.g_twobody, s.h_twobody);
end
fprintf('re-solved rule: f_lin %.9f  f_tb %.9f  diff %.3e  max dx %.3e\n', R.resolve.f_lin, ...
    R.resolve.f_tb, R.resolve.f_diff, R.resolve.max_abs_dx);
fprintf('naive Monte Carlo (no burn): %d hits in %d, pc %.3e, ci95 [%.2e %.2e], expected hits %.2f\n', ...
    R.naive.hits, R.naive.n, R.naive.pc, R.naive.ci95(1), R.naive.ci95(2), R.naive.expected_hits);
fprintf('propagator difference m: %.3e %.3e; Kepler round trip m: %.3e %.3e\n', ...
    R.prop_diff_m, R.kepler_roundtrip_m);
