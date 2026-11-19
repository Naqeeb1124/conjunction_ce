function test_viz_encounter()
% Checks the encounter-plane files and that the plotted stage values match the reference tables.
root = setup_paths();
check_animation_files('encounter_plane');

R = jsondecode(fileread(fullfile(root, 'results', 'viz_encounter_plane.json')));
E = load_expected();
cells = R.stage_cells;
idx = sub2ind([3 3], cells(:,1) + 1, cells(:,2) + 1);

check_close('encounter: stage xi vs miss_xi_table', R.stage_xi(:), E.baseline.miss_xi_table(idx), 1e-9);
check_close('encounter: stage pc vs pc_table', R.stage_pc(:), E.baseline.pc_table(idx), 1e-3);

[~, iPub] = max(E.baseline.schemes.public.table(:));
[pa1, pa2] = ind2sub([3 3], iPub);
check_true('encounter: stage 2 is the public-rule cell', ...
    isequal(cells(2,:), [pa1 pa2] - 1), sprintf('got [%d %d]', cells(2,1), cells(2,2)));
modal = E.baseline.ce_modal_recommendation(:)';
check_true('encounter: stage 3 is the modal recommendation', ...
    isequal(cells(3,:), modal), sprintf('got [%d %d]', cells(3,1), cells(3,2)));
end
