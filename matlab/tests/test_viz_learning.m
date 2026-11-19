function test_viz_learning()
% Checks the learning animation files and that the plotted numbers equal the source data (costs times 100).
root = setup_paths();
check_animation_files('learning');

R = jsondecode(fileread(fullfile(root, 'results', 'viz_learning.json')));
E = jsondecode(fileread(fullfile(root, 'expected.json')));
L = E.learning;

check_close('learning data: k', R.k, L.k, 1e-12, 0);
check_close('learning data: public', R.public, 100 * L.rules.public.coordinator_cost, 1e-12, 0);
check_close('learning data: private_m0', R.private_m0, 100 * L.rules.private_m0.coordinator_cost, 1e-12, 0);
check_close('learning data: private_uniform', R.private_uniform, 100 * L.rules.private_uniform.coordinator_cost, 1e-12, 0);
check_close('learning data: best_targeted', R.best_targeted, 100 * L.best_targeted.coordinator_cost, 1e-12, 0);
end
