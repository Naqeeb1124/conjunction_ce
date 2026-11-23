function save_final_png(fig, name)
root = setup_paths();
outDir = fullfile(root, 'figures');
if ~isfolder(outDir)
    mkdir(outDir);
end
print(fig, fullfile(outDir, [name '.png']), '-dpng', '-r200');
end
