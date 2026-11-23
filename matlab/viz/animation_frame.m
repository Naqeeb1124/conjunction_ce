function anim = animation_frame(anim, fig, repeat)
if nargin < 3 || isempty(repeat)
    repeat = 1;
end
frame = capture_frame(fig);
small = imresize(frame, [450 800]);
for r = 1:repeat
    writeVideo(anim.writer, frame);
    anim.n_frames = anim.n_frames + 1;
    if mod(anim.n_frames, 2) == 1
        anim.gif_frames{end+1} = small;
    end
end
end
