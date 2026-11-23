function info = animation_close(anim)
close(anim.writer);
mp4Path = fullfile(anim.root, 'figures', [anim.name '.mp4']);
gifPath = fullfile(anim.root, 'figures', [anim.name '.gif']);
limit = 7.5e6;
delay = 2 / anim.fps;
frames = anim.gif_frames;

bytes = write_gif_frames(frames, delay, gifPath);
if bytes >= limit
    frames = frames(1:2:end);
    delay = 2 * delay;
    bytes = write_gif_frames(frames, delay, gifPath);
    if bytes >= limit
        frames = cellfun(@(f) imresize(f, [360 640]), frames, 'UniformOutput', false);
        bytes = write_gif_frames(frames, delay, gifPath);
    end
end

info = struct('mp4', mp4Path, 'gif', gifPath, 'n_frames', anim.n_frames, 'gif_bytes', bytes);
fprintf('mp4: %s | gif: %s | n_frames: %d | gif_bytes: %d\n', mp4Path, gifPath, anim.n_frames, bytes);
end

function bytes = write_gif_frames(frames, delay, gifPath)
stack = vertcat(frames{end}, frames{ceil(numel(frames) / 2)});
[~, map] = rgb2ind(stack, 128, 'nodither');
for i = 1:numel(frames)
    ind = rgb2ind(frames{i}, map, 'nodither');
    if i == 1
        imwrite(ind, map, gifPath, 'gif', 'LoopCount', Inf, 'DelayTime', delay);
    else
        imwrite(ind, map, gifPath, 'gif', 'DelayTime', delay, 'WriteMode', 'append');
    end
end
d = dir(gifPath);
bytes = d.bytes;
end
