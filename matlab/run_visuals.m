function run_visuals()
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
