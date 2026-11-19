function test_viz_orbits()
% Checks the orbit animation files, the reference numbers and the plotted data in results/viz_orbits_3d.json.
root = setup_paths();
S = load_scenario();
E = load_expected();
R = jsondecode(fileread(fullfile(root, 'results', 'viz_orbits_3d.json')));

check_animation_files('orbits_3d');

rec = R.rec(:).';
u = S.burns(E.baseline.ce_modal_recommendation(:).' + 1);
u = u(:).';

check_close('orbits recommendation equals the reference', rec, ...
    E.baseline.ce_modal_recommendation(:).' + 1, 0, 0);
check_close('orbits no-burn miss equals the nominal miss', R.miss_A, ...
    hypot(S.miss_xi, S.miss_zeta), 0, 1e-3);
oB = two_body_miss(S, u(1), u(2));
check_close('orbits miss after the burns equals two_body_miss', R.miss_B, oB.miss, 0, 1e-3);
check_close('orbits minimum of the plotted distance equals the miss', ...
    [min(R.d_A_zoom) min(R.d_B_zoom)], [R.miss_A R.miss_B], 1e-9, 0);
check_close('orbits Pc of the two cases equals the reference table', [R.pc_A R.pc_B], ...
    [E.baseline.pc_table(1,1) E.baseline.pc_table(rec(1), rec(2))], 1e-3, 0);
check_close('orbits plotted positions stay on the orbits', [R.r_max R.r_min], ...
    [S.a S.a], 0, 100);
end
