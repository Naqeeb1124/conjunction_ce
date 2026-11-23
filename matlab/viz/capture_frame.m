function frame = capture_frame(fig)
drawnow;
try
    img = getframe(fig);
    frame = img.cdata;
catch
    tmp = [tempname '.png'];
    print(fig, tmp, '-dpng', '-r96');
    frame = imread(tmp);
    delete(tmp);
end
if ndims(frame) == 3 && size(frame, 3) > 3
    frame = frame(:, :, 1:3);
end
if ~isequal(size(frame, 1:2), [720 1280])
    frame = imresize(frame, [720 1280]);
end
frame = uint8(frame);
end
