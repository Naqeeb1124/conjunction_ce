function anim = animation_open(name, fps)
root = setup_paths();
outDir = fullfile(root, 'figures');
if ~isfolder(outDir)
    mkdir(outDir);
end
vw = VideoWriter(fullfile(outDir, [name '.mp4']), 'MPEG-4');
vw.FrameRate = fps;
vw.Quality = 95;
open(vw);
anim = struct('writer', vw, 'name', name, 'fps', fps, 'n_frames', 0, ...
    'gif_frames', {{}}, 'root', root);
end
