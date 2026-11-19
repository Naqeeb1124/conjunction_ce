function save_final_png(fig, name)
% Saves the figure as figures/<name>.png at 200 dpi (print, because exportgraphics crops).
root = setup_paths();
outDir = fullfile(root, 'figures');
if ~isfolder(outDir)
    mkdir(outDir);
end
print(fig, fullfile(outDir, [name '.png']), '-dpng', '-r200');
end
