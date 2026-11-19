function chk = run_simulink_check()
%RUN_SIMULINK_CHECK Compare the Simulink model with two_body_miss for the 9 joint actions.
%   chk = run_simulink_check() runs the model once per forced joint action, writes
%   results/simulink_check.json and the block-diagram image figures/conjunction_sim_model.png.
root = setup_paths();
S = load_scenario();
P = load_params();
mdl = build_conjunction_sim();
load_system(fullfile(root, 'simulink', mdl));
solver = get_param(mdl, 'Solver');
close_system(mdl, 0);

rows = repmat(struct('a1', 0, 'a2', 0, 'rec_a1', 0, 'rec_a2', 0, 'xi', 0, 'zeta', 0, 'miss', 0, ...
    'xi_ref', 0, 'zeta_ref', 0, 'd_xi', 0, 'd_zeta', 0, 'xi_axes_final', 0, 'zeta_axes_final', 0, 'predicted_rotation', 0, 'n_steps', 0), 9, 1);
n = 0;
for a1 = 1:3
    for a2 = 1:3
        n = n + 1;
        x = zeros(3);
        x(a1, a2) = 1;
        res = run_conjunction_sim(S, x, 0.5);
        ref = two_body_miss(S, S.burns(a1), S.burns(a2));
        rows(n) = struct('a1', a1, 'a2', a2, 'rec_a1', res.a1, 'rec_a2', res.a2, ...
            'xi', res.xi, 'zeta', res.zeta, 'miss', res.miss, 'xi_ref', ref.xi, 'zeta_ref', ref.zeta, ...
            'd_xi', res.xi - ref.xi, 'd_zeta', res.zeta - ref.zeta, 'xi_axes_final', res.xi_axes_final, ...
            'zeta_axes_final', res.zeta_axes_final, 'predicted_rotation', ...
            3 * S.T_lead * (S.burns(a2) - S.burns(a1)) / (2 * S.a * cos(S.phi / 2)) * res.xi, 'n_steps', res.n_steps);
    end
end
chk.solver = solver;
chk.reltol = P.two_body_check.ode_reltol;
chk.abstol = P.two_body_check.ode_abstol;
chk.atol_miss_m = 1e-3;
chk.n_steps = rows(1).n_steps;
chk.max_abs_d_xi = max(abs([rows.d_xi]));
chk.max_abs_d_zeta = max(abs([rows.d_zeta]));
chk.matlab_version = version;
sv = ver('simulink');
chk.simulink_version = sv.Version;
chk.runs = rows;

outdir = fullfile(root, 'results');
if ~exist(outdir, 'dir'), mkdir(outdir); end
fid = fopen(fullfile(outdir, 'simulink_check.json'), 'w');
fwrite(fid, jsonencode(chk, 'PrettyPrint', true));
fclose(fid);

% Block diagram image, model loaded without opening a window.
figdir = fullfile(root, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end
load_system(fullfile(root, 'simulink', mdl));
print(['-s' mdl], '-dpng', '-r200', fullfile(figdir, 'conjunction_sim_model.png'));
close_system(mdl, 0);
end
