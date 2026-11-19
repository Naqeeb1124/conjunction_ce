function run_visuals()
%RUN_VISUALS Render the four animations into figures/ (MP4, GIF and final-frame PNG each).
%   Every make_<name>_animation function also writes the numbers it plots to
%   results/viz_<name>.json, which tests/test_viz_<name>.m compares with their source.
setup_paths();
makers = {@make_encounter_animation, @make_orbits_animation, ...
    @make_gallery_animation, @make_learning_animation};
for i = 1:numel(makers)
    t0 = tic;
    makers{i}();
    close all force
    fprintf('%s: %.0f s\n', func2str(makers{i}), toc(t0));
end
end
