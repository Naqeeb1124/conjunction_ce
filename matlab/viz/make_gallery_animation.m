function out = make_gallery_animation()
% Renders the gallery of twelve conjunction geometries (MP4, GIF, final PNG) and writes results/viz_gallery.json.
root = setup_paths();
P = load_params();
st = viz_style();
nCase = numel(P.gallery);
nSweep = 150;
nHold = 75;

titleText = 'The coordinator''s recommendation in twelve conjunction geometries';
subtitleText = sprintf('%s\n%s', ...
    'Top view at closest approach. Hollow markers: nobody burns. Filled markers: after the recommended burns.', ...
    '(x%) = share of events with this recommendation.  (tie) = the swapped pair costs the coordinator the same.');
[fig, ax0] = viz_figure(titleText, subtitleText);
delete(ax0);
hs = findall(fig, '-property', 'String');
for h = hs(:)'
    s = get(h, 'String');
    if ischar(s) || iscell(s)
        txt = strjoin(cellstr(s), ' ');
        if contains(txt, 'Top view at closest approach')
            set(h, 'FontSize', 15);
        end
    end
end

colLeft = 0.035 + (0:3) * 0.238;
rowBottom = [0.615 0.353 0.091];
cellW = 0.205;
cellH = 0.150;
cellTextW = 0.236;

panels = cell(nCase, 1);
cases = repmat(struct('id', 0, 'a1', 0, 'a2', 0, 'prob', 0, 'tie', false, 'xi_after', 0, ...
    'pc_before', 0, 'pc_after', 0, 'miss_before', 0, 'miss_after', 0, 'tau_star', 0), nCase, 1);
for i = 1:nCase
    g = gallery_case(P, i);
    r = ceil(i / 4);
    c = mod(i - 1, 4) + 1;
    left = colLeft(c);
    bottom = rowBottom(r);

    ax = axes(fig, 'Units', 'normalized', 'Position', [left bottom cellW cellH]);
    H = build_panel(ax, g, st, nSweep);

    if g.S.lead_orbits == 1
        word = 'orbit';
    else
        word = 'orbits';
    end
    hdr = sprintf('%d   crossing %.0f deg, burn %g %s early', g.id, g.S.crossing_angle_deg, ...
        g.S.lead_orbits, word);
    line1 = sprintf('op 1: %s,  op 2: %s', lower(P.action_names{g.a1}), ...
        lower(P.action_names{g.a2}));
    if g.tie
        line1 = [line1 '  (tie)']; %#ok<AGROW>
    elseif g.prob < 1 - 1e-9
        line1 = [line1 sprintf('  (%.0f%%)', 100 * g.prob)]; %#ok<AGROW>
    end
    line2 = sprintf('pass %.0f m -> %.0f m,  Pc %s -> %s', g.miss_before, g.miss_after, ...
        sprintf('%.1e', g.pc_before), sprintf('%.1e', g.pc_after));

    annotation(fig, 'textbox', [left, bottom + cellH + 0.005, cellTextW, 0.028], ...
        'String', hdr, 'EdgeColor', 'none', 'FitBoxToText', 'off', ...
        'FontName', st.font_title, 'FontSize', 11, 'Color', st.ink, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Interpreter', 'none');
    annotation(fig, 'textbox', [left, bottom - 0.031, cellTextW, 0.026], 'String', line1, ...
        'EdgeColor', 'none', 'FitBoxToText', 'off', 'FontName', st.font, 'FontSize', 10.5, ...
        'Color', st.ink, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', ...
        'Interpreter', 'none');
    H.hLine2 = annotation(fig, 'textbox', [left, bottom - 0.057, cellTextW, 0.026], ...
        'String', '', 'EdgeColor', 'none', 'FitBoxToText', 'off', 'FontName', st.font, ...
        'FontSize', 10.5, 'Color', st.ink2, 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', 'Interpreter', 'none');
    H.line2 = line2;
    panels{i} = H;

    cases(i).id = g.id;
    cases(i).a1 = g.a1 - 1;
    cases(i).a2 = g.a2 - 1;
    cases(i).prob = g.prob;
    cases(i).tie = g.tie;
    cases(i).xi_after = g.xi_after;
    cases(i).pc_before = g.pc_before;
    cases(i).pc_after = g.pc_after;
    cases(i).miss_before = g.miss_before;
    cases(i).miss_after = g.miss_after;
    cases(i).tau_star = g.tau_star;
end

anim = animation_open('gallery', st.fps);
for k = 1:nSweep
    for i = 1:nCase
        set_panel_frame(panels{i}, k);
    end
    rep = 1;
    if k == nSweep
        rep = 1 + nHold;
    end
    anim = animation_frame(anim, fig, rep);
end
save_final_png(fig, 'gallery');
info = animation_close(anim);

R.n_frames = info.n_frames;
R.cases = cases;
if ~isfolder(fullfile(root, 'results'))
    mkdir(fullfile(root, 'results'));
end
fid = fopen(fullfile(root, 'results', 'viz_gallery.json'), 'w');
fprintf(fid, '%s', jsonencode(R));
fclose(fid);

close(fig);
out = R;
end

function H = build_panel(ax, g, st, nSweep)
% Draws the static parts of one panel and creates the objects that the frames update.
Si = g.S;
v = Si.v;
half = g.half;
e1 = [cos(half), -sin(half)];
e2 = [cos(half), sin(half)];
d1 = g.d1;
d2 = g.d2;
xi0 = g.xi0;

pb = getpixelposition(ax);
ratio = pb(3) / pb(4);
cx = xi0 / 2;
cy = (d1 + d2) / 4 * sin(half);
hh = 1.25 * max([2.2 * Si.sigma_xi, abs(g.xi_after), 0.6 * (d1 + d2)]);
hw = hh * ratio;
xl = cx + [-hw hw];
yl = cy + [-hh hh];
set(ax, 'NextPlot', 'add', 'XLim', xl, 'YLim', yl, 'XLimMode', 'manual', 'YLimMode', 'manual', ...
    'XTick', [], 'YTick', [], 'Box', 'on', 'XColor', st.grid, 'YColor', st.grid, ...
    'Color', 'w', 'XGrid', 'off', 'YGrid', 'off', 'LineWidth', 0.8, 'Clipping', 'on');

tauw = 1.5 * max(hw, hh) / v;
tau = linspace(-tauw, tauw, nSweep);
p1 = [v * tau * e1(1); v * tau * e1(2)];
p2 = [xi0 + v * tau * e2(1); v * tau * e2(2)];
q1 = [(v * tau - d1) * e1(1); (v * tau - d1) * e1(2)];
q2 = [xi0 + (v * tau + d2) * e2(1); (v * tau + d2) * e2(2)];
ts = g.tau_star;
qs1 = [(v * ts - d1) * e1(1); (v * ts - d1) * e1(2)];
qs2 = [xi0 + (v * ts + d2) * e2(1); (v * ts + d2) * e2(2)];

ellCols = {st.ell3, st.ell2, st.ell1};
hE = gobjects(3, 1);
for n = 1:3
    hE(n) = rectangle('Parent', ax, 'Position', ...
        [qs1(1) - n * Si.sigma_xi, qs1(2) - n * g.sig_s, 2 * n * Si.sigma_xi, 2 * n * g.sig_s], ...
        'Curvature', [1 1], 'FaceColor', ellCols{n}, 'EdgeColor', 'none', 'Visible', 'off');
end

Lt = 3 * (hw + hh + abs(xi0) + abs(d1) + abs(d2));
plot(ax, Lt * [-1 1] * e1(1), Lt * [-1 1] * e1(2), '-', 'Color', mix_white(st.blue), 'LineWidth', 1.2);
plot(ax, xi0 + Lt * [-1 1] * e2(1), Lt * [-1 1] * e2(2), '-', 'Color', mix_white(st.orange), 'LineWidth', 1.2);

hDash = plot(ax, [0 xi0], [0 0], '--', 'Color', st.muted, 'LineWidth', 1.2, 'Visible', 'off');
hInk = plot(ax, [qs1(1) qs2(1)], [qs1(2) qs2(2)], '-', 'Color', st.ink, 'LineWidth', 1.6, ...
    'Visible', 'off');

hH1 = plot(ax, NaN, NaN, 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', st.muted, 'LineWidth', 1.2, 'LineStyle', 'none');
hH2 = plot(ax, NaN, NaN, 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', st.muted, 'LineWidth', 1.2, 'LineStyle', 'none');

hT1 = plot(ax, NaN, NaN, '-', 'Color', st.blue, 'LineWidth', 2.2);
hT2 = plot(ax, NaN, NaN, '-', 'Color', st.orange, 'LineWidth', 2.2);

hF1 = plot(ax, NaN, NaN, 'o', 'MarkerSize', 8, 'MarkerFaceColor', st.blue, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1, 'LineStyle', 'none');
hF2 = plot(ax, NaN, NaN, 'o', 'MarkerSize', 8, 'MarkerFaceColor', st.orange, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1, 'LineStyle', 'none');
hS1 = plot(ax, qs1(1), qs1(2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', st.blue, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1, 'LineStyle', 'none', 'Visible', 'off');
hS2 = plot(ax, qs2(1), qs2(2), 'o', 'MarkerSize', 8, 'MarkerFaceColor', st.orange, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1, 'LineStyle', 'none', 'Visible', 'off');

lens = [50 100 200 500 1000 2000];
ok = lens(lens <= 0.35 * hw);
if isempty(ok)
    Lbar = 50;
else
    Lbar = max(ok);
end
x0 = xl(1) + 0.12 * hw;
y0 = yl(1) + 0.18 * hh;
plot(ax, [x0 x0 + Lbar], [y0 y0], '-', 'Color', st.ink, 'LineWidth', 1.6);
text(ax, x0, y0 + 0.06 * hh, sprintf('%d m', Lbar), 'FontName', st.font, 'FontSize', 10, ...
    'Color', st.ink, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
    'Interpreter', 'none');

H.tau = tau;
H.p1 = p1;
H.p2 = p2;
H.q1 = q1;
H.q2 = q2;
H.xl = xl;
H.yl = yl;
H.xi0 = xi0;
H.tau_star = ts;
H.nobody = g.nobody;
H.hH1 = hH1;
H.hH2 = hH2;
H.hDash = hDash;
H.hT1 = hT1;
H.hT2 = hT2;
H.hF1 = hF1;
H.hF2 = hF2;
H.hS1 = hS1;
H.hS2 = hS2;
H.hInk = hInk;
H.hE = hE;
end

function set_panel_frame(H, k)
% Moves the markers of one panel to frame k and switches on the stamps once tau has passed them.
t = H.tau(k);
if ~H.nobody
    if t < 0
        a = H.p1(:, k);
        b = H.p2(:, k);
        set(H.hH1, 'XData', a(1), 'YData', a(2), 'Visible', onoff(in_panel(H, a)));
        set(H.hH2, 'XData', b(1), 'YData', b(2), 'Visible', onoff(in_panel(H, b)));
        set(H.hDash, 'Visible', 'off');
    else
        set(H.hH1, 'XData', 0, 'YData', 0, 'Visible', 'on');
        set(H.hH2, 'XData', H.xi0, 'YData', 0, 'Visible', 'on');
        set(H.hDash, 'Visible', 'on');
    end
end

a = H.q1(:, k);
b = H.q2(:, k);
set(H.hF1, 'XData', a(1), 'YData', a(2), 'Visible', onoff(in_panel(H, a)));
set(H.hF2, 'XData', b(1), 'YData', b(2), 'Visible', onoff(in_panel(H, b)));
idx = max(1, k - 11):k;
set(H.hT1, 'XData', H.q1(1, idx), 'YData', H.q1(2, idx));
set(H.hT2, 'XData', H.q2(1, idx), 'YData', H.q2(2, idx));

passed = t >= H.tau_star;
if passed ~= strcmp(get(H.hS1, 'Visible'), 'on')
    set([H.hS1, H.hS2, H.hInk], 'Visible', onoff(passed));
    set(H.hE, 'Visible', onoff(passed));
    if passed
        set(H.hLine2, 'String', H.line2);
    else
        set(H.hLine2, 'String', '');
    end
end
end

function tf = in_panel(H, p)
% True when the point p lies inside the panel limits.
tf = p(1) >= H.xl(1) && p(1) <= H.xl(2) && p(2) >= H.yl(1) && p(2) <= H.yl(2);
end

function s = onoff(tf)
% Maps a logical to the Visible property value.
if tf
    s = 'on';
else
    s = 'off';
end
end

function c = mix_white(col)
% Mixes a colour with 55 % white.
c = 0.45 * col + 0.55;
end
