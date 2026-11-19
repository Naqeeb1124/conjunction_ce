function test_simulink()
%TEST_SIMULINK Checks of the Simulink conjunction model against the two-body reference.
root = setup_paths();
S = load_scenario();

% a. The model is created by the build script.
slx = fullfile(root, 'simulink', 'conjunction_sim.slx');
if bdIsLoaded('conjunction_sim'), close_system('conjunction_sim', 0); end
if exist(slx, 'file'), delete(slx); end
mdl = build_conjunction_sim();
load_system(slx);
subs = get_param(find_system(mdl, 'SearchDepth', 1, 'BlockType', 'SubSystem'), 'Name');
want = {'Coordinator', 'Burn', 'Satellite 1', 'Satellite 2', 'Miss distance'};
check_true('simulink model is built by the script', ...
    isfile(slx) && all(ismember(want, subs)), 'file exists, 5 subsystems');
close_system(mdl, 0);

% b. Forced joint actions.
forced = true;
got = zeros(9, 2);
ref = zeros(9, 2);
res_all = cell(9, 1);
rot_got = zeros(9, 1);
rot_want = zeros(9, 1);
n = 0;
for a1 = 1:3
    for a2 = 1:3
        n = n + 1;
        x = zeros(3);
        x(a1, a2) = 1;
        r = run_conjunction_sim(S, x, 0.5);
        res_all{n} = r;
        forced = forced && r.a1 == a1 && r.a2 == a2;
        out = two_body_miss(S, S.burns(a1), S.burns(a2));
        got(n, :) = [r.xi r.zeta];
        eta = 3 * S.T_lead * (S.burns(a2) - S.burns(a1)) / (2 * S.a * cos(S.phi / 2));
        rot_got(n) = r.zeta_axes_final - r.zeta;
        rot_want(n) = eta * r.xi;
        ref(n, :) = [out.xi out.zeta];
    end
end
check_true('simulink recommends the forced joint action in all 9 runs', forced, '9 runs');
check_close('simulink miss vector vs two_body_miss, 9 joint actions', got, ref, 0, 1e-3);

% c. Inverse-CDF draw with the coordinator's rule.
T = cost_matrices(S);
x_rule = solve_ce(T.c1, T.c2, T.obj);
cdf = cumsum(reshape(x_rule.', [], 1));
ok = true;
for u = [0.05 0.25 0.5 0.75 0.95]
    k = find(cdf > u, 1);
    if isempty(k), k = 9; end
    r = run_conjunction_sim(S, x_rule, u);
    ok = ok && r.a1 == floor((k - 1) / 3) + 1 && r.a2 == mod(k - 1, 3) + 1;
end
check_true('simulink coordinator draws by inverse CDF', ok, 'u = 0.05 ... 0.95');

check_close('simulink axes from the velocities at the nominal time are turned by (d1+d2)/(2 a cos(phi/2))', rot_got, rot_want, 0, 1e-3);

% d. No burn gives the nominal miss.
check_close('simulink no-burn run returns the nominal miss', got(1, :), [S.miss_xi S.miss_zeta], 0, 1e-3);

% e. Range history ends consistently with the straight-line closest approach.
lhs = zeros(9, 1);
rhs = zeros(9, 1);
for n = 1:9
    r = res_all{n};
    lhs(n) = r.range(end)^2;
    rhs(n) = r.miss^2 + (norm(r.w_final) * r.t_ca)^2;
end
check_close('simulink final range is consistent with the miss vector', lhs, rhs, 1e-6, 0);

% Write results/simulink_check.json and the diagram image.
run_simulink_check();
end
