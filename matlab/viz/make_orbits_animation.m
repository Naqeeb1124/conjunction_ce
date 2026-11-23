function out = make_orbits_animation()
root = setup_paths();
S = load_scenario();
E = load_expected();
st = viz_style();

rec = E.baseline.ce_modal_recommendation(:).' + 1;
u = S.burns(rec);
u = u(:).';
oA = two_body_miss(S, 0, 0, 'kepler');
oB = two_body_miss(S, u(1), u(2), 'kepler');
pcA = pc2d(oA.xi, oA.zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
pcB = pc2d(oB.xi, oB.zeta, S.sigma_xi, S.sigma_zeta, S.hbr);

T = S.T;
tz = 0.2;

tauZ = (-200:200) * (tz / 200);
tauL = (-2000:500) * (T / 8000);
dZA = separation(oA, S, tauZ);
dZB = separation(oB, S, tauZ);
dLA = separation(oA, S, tauL);
dLB = separation(oB, S, tauL);

s = linspace(0, 1, 170);
tauApp = -tz - (T/4 - tz) * (1 - s).^2;
tauSlow = linspace(-tz, tz, 90);
tauDep = linspace(tz, T/16, 50);
tauAll = [tauApp, tauSlow, tauDep];
nFrames = numel(tauAll) - 1 + 60;

titleText = sprintf('The recommended burns widen the pass from %.0f m to %.0f m', oA.miss, oB.miss);
subtitleText = sprintf(['Both satellites at %.0f km altitude. Burns of %g and %g cm/s, ' ...
    '%g orbits before closest approach.'], S.altitude/1e3, 100*u(1), 100*u(2), S.lead_orbits);
[fig, axDefault] = viz_figure(titleText, subtitleText);
delete(axDefault);

ax3 = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.12 0.16 0.38 0.58]);
hold(ax3, 'on');
[Xe, Ye, Ze] = sphere(36);
surf(ax3, S.r_earth*Xe, S.r_earth*Ye, S.r_earth*Ze, 'FaceColor', [0.93 0.95 0.97], ...
    'EdgeColor', st.grid, 'FaceLighting', 'none', 'EdgeLighting', 'none');

th = linspace(0, 2*pi, 361);
R1 = ringPoints(oA.r1, oA.v1, th);
R2 = ringPoints(oA.r2, oA.v2, th);
plot3(ax3, R1(1,:), R1(2,:), R1(3,:), '-', 'Color', st.blue, 'LineWidth', 1.6);
plot3(ax3, R2(1,:), R2(2,:), R2(3,:), '-', 'Color', st.orange, 'LineWidth', 1.6);

pX = 0.5 * (oA.r1(:) + oA.r2(:));
nX = pX / norm(pX);
plot3(ax3, pX(1), pX(2), pX(3), 'o', 'MarkerSize', 9, 'MarkerFaceColor', 'none', ...
    'MarkerEdgeColor', st.ink, 'LineWidth', 1.2, 'LineStyle', 'none');
axis(ax3, 'equal');
axis(ax3, 'off');
axis(ax3, 'vis3d');
view(ax3, [2 -1.3 0.5]);
camzoom(ax3, 1.28);

hTr1 = plot3(ax3, NaN, NaN, NaN, '-', 'Color', st.blue, 'LineWidth', 3.4);
hTr2 = plot3(ax3, NaN, NaN, NaN, '-', 'Color', st.orange, 'LineWidth', 3.4);
hS1 = plot3(ax3, NaN, NaN, NaN, 'o', 'MarkerSize', 11, 'MarkerFaceColor', st.blue, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');
hS2 = plot3(ax3, NaN, NaN, NaN, 'o', 'MarkerSize', 11, 'MarkerFaceColor', st.orange, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');

hRead = annotation(fig, 'textbox', [0.02 0.755 0.47 0.05], 'String', '', ...
    'EdgeColor', 'none', 'FitBoxToText', 'off', 'FontName', st.font, ...
    'FontSize', st.fs_label, 'Color', st.ink, 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'bottom', 'Interpreter', 'none');
keyTxt = {[char(9679) '  satellite 1'], [char(9679) '  satellite 2'], [char(9675) '  closest approach']};
keyCol = {st.blue, st.orange, st.ink};
for kk = 1:3
    annotation(fig, 'textbox', [0.02 0.650 - 0.038*(kk-1) 0.20 0.034], 'String', keyTxt{kk}, ...
        'EdgeColor', 'none', 'BackgroundColor', 'none', 'FitBoxToText', 'off', ...
        'FontName', st.font, 'FontSize', st.fs_note, 'Color', keyCol{kk}, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Interpreter', 'none');
end
annotation(fig, 'textbox', [0.02 0.06 0.52 0.05], 'EdgeColor', 'none', 'FitBoxToText', 'off', ...
    'String', sprintf('Drawn to scale: the burns move the pass point by %.0f m, too little to see here.', ...
    S.gain*(u(1)+u(2))), 'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.ink2, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Interpreter', 'none');

axT = axes('Parent', fig, 'Units', 'normalized');
viz_axes(axT);
hold(axT, 'on');
set(axT, 'Position', [0.62 0.57 0.35 0.22]);
xMinT = -T/4/60;
xMaxT = T/16/60;
xt = ceil(xMinT/5)*5 : 5 : floor(xMaxT/5)*5;
set(axT, 'XLim', [xMinT xMaxT], 'XTick', xt, ...
    'XTickLabel', arrayfun(@(v) sprintf('%d', v), xt, 'UniformOutput', false), ...
    'YScale', 'log', 'YLim', [10 2e7], 'YTick', [1e2 1e3 1e4 1e5 1e6 1e7], ...
    'YTickLabel', {'100 m', '1 km', '10 km', '100 km', '1000 km', '10 000 km'});
xlabel(axT, 'minutes from closest approach', 'FontName', st.font, 'FontSize', st.fs_label);
title(axT, 'Distance between the satellites', 'FontName', st.font, ...
    'FontSize', st.fs_panel, 'FontWeight', 'normal', 'Color', st.ink);
hTA = plot(axT, NaN, NaN, '-', 'Color', st.ink2, 'LineWidth', st.lw);
hTB = plot(axT, NaN, NaN, '-', 'Color', st.aqua, 'LineWidth', st.lw);

axB = axes('Parent', fig, 'Units', 'normalized');
viz_axes(axB);
hold(axB, 'on');
yTopB = 1.05 * max([dZA dZB]) / 1000;
set(axB, 'Position', [0.62 0.13 0.35 0.22], 'XLim', [-tz tz], 'YLim', [0 yTopB], ...
    'XTick', [-0.2 -0.1 0 0.1 0.2], 'XTickLabel', {'-0.2', '-0.1', '0', '0.1', '0.2'});
xlabel(axB, 'seconds from closest approach', 'FontName', st.font, 'FontSize', st.fs_label);
ylabel(axB, 'distance (km)', 'FontName', st.font, 'FontSize', st.fs_label);
title(axB, sprintf('The last %.1f seconds', 2*tz), 'FontName', st.font, ...
    'FontSize', st.fs_panel, 'FontWeight', 'normal', 'Color', st.ink);
hBA = plot(axB, NaN, NaN, '-', 'Color', st.ink2, 'LineWidth', st.lw);
hBB = plot(axB, NaN, NaN, '-', 'Color', st.aqua, 'LineWidth', st.lw);
hMinA = plot(axB, NaN, NaN, 'o', 'MarkerSize', 9, 'MarkerFaceColor', st.ink2, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');
hMinB = plot(axB, NaN, NaN, 'o', 'MarkerSize', 9, 'MarkerFaceColor', st.aqua, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');
hLabA = text(axB, -0.065, 0.45,sprintf('nobody burns:\n%.0f m, Pc %.1e', oA.miss, pcA), ...
    'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.ink2, ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', ...
    'Interpreter', 'none', 'Visible', 'off');
hLabB = text(axB, 0.065, 0.45,sprintf('after the burns:\n%.0f m, Pc %.1e', oB.miss, pcB), ...
    'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.aqua, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
    'Interpreter', 'none', 'Visible', 'off');

rmax = 0;
rmin = Inf;
anim = animation_open('orbits_3d', st.fps);
for i = 1:numel(tauAll)
    tn = tauAll(i);
    tt = linspace(tn - T/8, tn, 40);
    P1 = track(oA.r1, oA.v1, tt, S.mu);
    P2 = track(oA.r2, oA.v2, tt, S.mu);
    rmax = max([rmax max(vecnorm(P1)) max(vecnorm(P2))]);
    rmin = min([rmin min(vecnorm(P1)) min(vecnorm(P2))]);
    set(hTr1, 'XData', P1(1,:), 'YData', P1(2,:), 'ZData', P1(3,:));
    set(hTr2, 'XData', P2(1,:), 'YData', P2(2,:), 'ZData', P2(3,:));
    set(hS1, 'XData', P1(1,end), 'YData', P1(2,end), 'ZData', P1(3,end));
    set(hS2, 'XData', P2(1,end), 'YData', P2(2,end), 'ZData', P2(3,end));

    dAn = norm(P2(:,end) - P1(:,end));
    dBn = separation(oB, S, tn);
    if tn < 0
        secs = round(-tn);
        set(hRead, 'String', sprintf('%d min %02d s to closest approach', floor(secs/60), mod(secs, 60)));
    else
        set(hRead, 'String', 'after closest approach');
    end

    [x, y] = grownCurve(tauL/60, dLA, tn/60, dAn);
    set(hTA, 'XData', x, 'YData', y);
    [x, y] = grownCurve(tauL/60, dLB, tn/60, dBn);
    set(hTB, 'XData', x, 'YData', y);

    [x, y] = grownCurve(tauZ, dZA/1000, tn, dAn/1000);
    set(hBA, 'XData', x, 'YData', y);
    [x, y] = grownCurve(tauZ, dZB/1000, tn, dBn/1000);
    set(hBB, 'XData', x, 'YData', y);

    rep = 1;
    if i == numel(tauAll)
        rep = 60;
        set(hMinA, 'XData', 0, 'YData', oA.miss/1000);
        set(hMinB, 'XData', 0, 'YData', oB.miss/1000);
        set([hLabA hLabB], 'Visible', 'on');
    end
    anim = animation_frame(anim, fig, rep);
end

save_final_png(fig, 'orbits_3d');
info = animation_close(anim);

R.rec = rec;
R.u = u;
R.miss_A = oA.miss;
R.miss_B = oB.miss;
R.xi_A = oA.xi;
R.zeta_A = oA.zeta;
R.xi_B = oB.xi;
R.zeta_B = oB.zeta;
R.pc_A = pcA;
R.pc_B = pcB;
R.tau_zoom = tauZ(:);
R.d_A_zoom = dZA(:);
R.d_B_zoom = dZB(:);
R.tau_long = tauL(:);
R.d_A_long = dLA(:);
R.d_B_long = dLB(:);
R.r_max = rmax;
R.r_min = rmin;
R.n_frames = nFrames;
R.file_info = struct('mp4', 'figures/orbits_3d.mp4', 'gif', 'figures/orbits_3d.gif', ...
    'n_frames', info.n_frames, 'gif_bytes', info.gif_bytes);

if ~isfolder(fullfile(root, 'results'))
    mkdir(fullfile(root, 'results'));
end
fid = fopen(fullfile(root, 'results', 'viz_orbits_3d.json'), 'w');
fprintf(fid, '%s', jsonencode(R));
fclose(fid);

close(fig);
out = R;
end

function R = ringPoints(r, v, th)
R = norm(r) * (r(:) / norm(r) * cos(th) + v(:) / norm(v) * sin(th));
end

function P = track(r0, v0, tau, mu)
P = zeros(3, numel(tau));
for k = 1:numel(tau)
    p = propagate_kepler(r0, v0, tau(k), mu);
    P(:, k) = p(:);
end
end

function d = separation(o, S, tau)
d = zeros(size(tau));
for k = 1:numel(tau)
    r1 = propagate_kepler(o.r1, o.v1, tau(k), S.mu);
    r2 = propagate_kepler(o.r2, o.v2, tau(k), S.mu);
    d(k) = norm(r2(:) - r1(:));
end
end

function [x, y] = grownCurve(tg, yg, tn, yn)
m = tg <= tn;
x = tg(m);
y = yg(m);
if tn >= tg(1) && tn <= tg(end) && (isempty(x) || tn > x(end))
    x(end+1) = tn;
    y(end+1) = yn;
end
end
