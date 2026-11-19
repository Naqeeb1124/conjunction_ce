function check_animation_files(name)
% Checks figures/<name>.mp4, .gif and .png: presence, GIF size, video dimensions and PNG width.
root = setup_paths();
mp4 = fullfile(root, 'figures', [name '.mp4']);
gif = fullfile(root, 'figures', [name '.gif']);
png = fullfile(root, 'figures', [name '.png']);

check_true([name ': mp4, gif and png exist'], isfile(mp4) && isfile(gif) && isfile(png), ...
    sprintf('%s | %s | %s', mp4, gif, png));

gifBytes = 0;
if isfile(gif)
    d = dir(gif);
    gifBytes = d.bytes;
end
check_true([name ': gif under 8e6 bytes'], gifBytes < 8e6, sprintf('%d bytes', gifBytes));

w = 0; h = 0; nf = 0;
if isfile(mp4)
    vr = VideoReader(mp4);
    w = vr.Width;
    h = vr.Height;
    nf = vr.NumFrames;
end
check_true([name ': mp4 1280x720 with at least 60 frames'], ...
    w == 1280 && h == 720 && nf >= 60, sprintf('%dx%d, %d frames', w, h, nf));

pw = 0;
if isfile(png)
    info = imfinfo(png);
    pw = info.Width;
end
check_true([name ': png at least 1920 px wide'], pw >= 1920, sprintf('%d px wide', pw));
end
