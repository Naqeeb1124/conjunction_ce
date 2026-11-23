function out = make_encounter_animation()
root = setup_paths();
S = load_scenario();
E = load_expected();
P = load_params();
st = viz_style();

Tpub = E.baseline.schemes.public.table;
[~, iPub] = max(Tpub(:));
[pa1, pa2] = ind2sub(size(Tpub), iPub);
modal = E.baseline.ce_modal_recommendation(:) + 1;
cells = [1 1; pa1 pa2; modal(1) modal(2)];

u = S.burns(:);
uStage = [u(cells(:,1)), u(cells(:,2))];
xiS = zeros(3, 1);
pcS = zeros(3, 1);
for k = 1:3
    xiS(k) = miss_shift(S, uStage(k,1), uStage(k,2));
    pcS(k) = pc2d(xiS(k), S.miss_zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
end
pcFloor = 1.1e-10;
zeta = S.miss_zeta;
pc0 = pcS(1);

stageLabel = cell(1, 3);
stageLabel{1} = 'nobody burns';
for k = 2:3
    a = cells(k,:) - 1;
    if a(1) == 0 && a(2) == 0
        stageLabel{k} = 'nobody burns';
    elseif a(1) > 0 && a(2) == 0
        stageLabel{k} = sprintf('operator 1: %s', P.action_names{a(1)+1});
    elseif a(1) == 0 && a(2) > 0
        stageLabel{k} = sprintf('operator 2: %s', P.action_names{a(2)+1});
    elseif a(1) == a(2)
        stageLabel{k} = sprintf('both: %s', P.action_names{a(1)+1});
    else
        stageLabel{k} = sprintf('operator 1: %s, operator 2: %s', ...
            P.action_names{a(1)+1}, P.action_names{a(2)+1});
    end
end

titleText = 'Each burn slides the uncertainty ellipse away from satellite 1';
subtitleText = sprintf('Encounter plane at closest approach: a 1 cm/s burn moves the pass point %.0f m.', S.gain*0.01);
[fig, ax] = viz_figure(titleText, subtitleText);

set(ax, 'Position', [0.075 0.13 0.555 0.60], 'XLim', [-420 1000], 'YLim', [-175 205]);
xlabel(ax, '\xi: along the bisector of the two velocity directions (m)', 'Interpreter', 'tex');
ylabel(ax, '\zeta: radial (m)', 'Interpreter', 'tex');
set([ax.XLabel, ax.YLabel], 'FontName', st.font, 'FontSize', st.fs_label, 'Color', st.ink);
text(ax, 0, 1.03, 'Predicted pass of satellite 2 and its uncertainty', 'Units', 'normalized', ...
    'Clipping', 'off', 'FontName', st.font_title, 'FontSize', st.fs_panel, 'Color', st.ink, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Interpreter', 'none');

th = linspace(0, 2*pi, 240);
ns = [3 2 1];
ellCols = {st.ell3, st.ell2, st.ell1};
hE = gobjects(1, 3);
for q = 1:3
    hE(q) = patch(ax, xiS(1) + ns(q)*S.sigma_xi*cos(th), zeta + ns(q)*S.sigma_zeta*sin(th), ellCols{q}, ...
        'EdgeColor', 'w', 'LineWidth', 2);
end

hTrail = plot(ax, [xiS(1) xiS(1)], [zeta zeta], '-', 'Color', st.axis, 'LineWidth', 1.6);

dyLabel = 20;
hSM = gobjects(1, 3);
hSL = gobjects(1, 3);
for k = 1:3
    hSM(k) = plot(ax, xiS(k), zeta, 'o', 'MarkerSize', 12, 'MarkerFaceColor', st.orange, ...
        'MarkerEdgeColor', 'w', 'LineStyle', 'none', 'Visible', 'off');
    if k == 2
        hSL(k) = text(ax, xiS(k), zeta - dyLabel, stageLabel{k}, 'FontName', st.font, ...
            'FontSize', st.fs_note, 'Color', st.ink, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', 'Interpreter', 'none', 'Visible', 'off');
    else
        hSL(k) = text(ax, xiS(k), zeta + dyLabel, stageLabel{k}, 'FontName', st.font, ...
            'FontSize', st.fs_note, 'Color', st.ink, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'Interpreter', 'none', 'Visible', 'off');
    end
end

hPass = plot(ax, xiS(1), zeta, 'o', 'MarkerSize', 12, 'MarkerFaceColor', st.orange, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');

plot(ax, 0, 0, 'o', 'MarkerSize', 7, 'MarkerFaceColor', st.ink, 'MarkerEdgeColor', st.ink, ...
    'LineStyle', 'none');
plot(ax, [0 -400], [0 -60], '-', 'Color', st.muted, 'LineWidth', 1);
text(ax, -400, -110, sprintf('satellite 1 at the origin\n(combined hard-body radius %g m)', S.hbr), ...
    'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.ink2, 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'bottom', 'Interpreter', 'none');

hRead = text(ax, -400, 200, '', 'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.ink, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'Interpreter', 'none');
text(ax, 995, -170, 'ellipses: 1, 2, 3 sigma of the combined position uncertainty (vertical scale stretched)', ...
    'FontName', st.font, 'FontSize', 13, 'Color', st.muted, 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'bottom', 'Interpreter', 'none');

axR = axes(fig, 'Position', [0.745 0.13 0.20 0.60]);
viz_axes(axR);
set(axR, 'YScale', 'log', 'YLim', [1e-10 1e-2], 'YTick', 10.^(-10:2:-2), ...
    'XLim', [0 1], 'XTick', [], 'XGrid', 'off');
text(axR, 0, 1.03, 'Collision probability', 'Units', 'normalized', 'Clipping', 'off', ...
    'FontName', st.font_title, 'FontSize', st.fs_panel, 'Color', st.ink, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Interpreter', 'none');
hCount = text(axR, 0, 1.10, '', 'Units', 'normalized', 'Clipping', 'off', ...
    'FontName', st.font_title, 'FontSize', 22, 'Color', st.ink, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Interpreter', 'tex');
hSub = text(axR, 0.04, 0.97, '', 'Units', 'normalized', 'FontName', st.font, ...
    'FontSize', st.fs_note, 'Color', st.ink2, 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'Interpreter', 'none');
plot(axR, [0 1], [S.pc_threshold S.pc_threshold], '-', 'Color', st.ink2, 'LineWidth', 1.4);
text(axR, 0.53, S.pc_threshold, sprintf('threshold 10^{%d}', round(log10(S.pc_threshold))), ...
    'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.ink2, 'Interpreter', 'tex', ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Clipping', 'off', 'Margin', 1);
hVL = plot(axR, [0.5 0.5], [pc0 pc0], '-', 'Color', st.axis, 'LineWidth', 1.6);
hPc = plot(axR, 0.5, pc0, 'o', 'MarkerSize', 12, 'MarkerFaceColor', st.orange, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');

B = zeros(0, 4);
B = [B; uStage(1,:) 45 1];
for j = 1:90
    w = 0.5 - 0.5*cos(pi*j/90);
    B = [B; uStage(1,:) + w*(uStage(2,:) - uStage(1,:)) 1 2*(j == 90)];
end
B = [B; uStage(2,:) 45 0];
for j = 1:90
    w = 0.5 - 0.5*cos(pi*j/90);
    B = [B; uStage(2,:) + w*(uStage(3,:) - uStage(2,:)) 1 3*(j == 90)];
end
B = [B; uStage(3,:) 75 0];

anim = animation_open('encounter_plane', st.fps);
for r = 1:size(B, 1)
    xi = miss_shift(S, B(r,1), B(r,2));
    pc = pc2d(xi, zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
    plotPc = max(pc, pcFloor);

    for q = 1:3
        set(hE(q), 'XData', xi + ns(q)*S.sigma_xi*cos(th), 'YData', zeta + ns(q)*S.sigma_zeta*sin(th));
    end
    set(hTrail, 'XData', [xiS(1) xi], 'YData', [zeta zeta]);
    set(hPass, 'XData', xi, 'YData', zeta);
    k = B(r,4);
    if k > 0
        set(hSM(k), 'Visible', 'on');
        set(hSL(k), 'Visible', 'on');
    end

    set(hPc, 'YData', plotPc);
    set(hVL, 'YData', [max(pc0, pcFloor) plotPc]);
    if pc > 0
        ex = floor(log10(pc));
        mant = pc / 10^ex;
        if round(mant, 1) >= 10
            mant = mant / 10;
            ex = ex + 1;
        end
        counterStr = sprintf('P_c = %.1f \\times 10^{%d}', mant, ex);
    else
        counterStr = 'P_c = 0';
    end
    set(hCount, 'String', counterStr);
    if pc > S.pc_threshold
        set(hSub, 'String', 'above the threshold');
    else
        set(hSub, 'String', 'below the threshold');
    end
    set(hRead, 'String', sprintf(['operator 1 burn: %.1f cm/s     operator 2 burn: %.1f cm/s\n' ...
        'pass point moved: %.0f m'], 100*B(r,1), 100*B(r,2), xi - S.miss_xi));

    anim = animation_frame(anim, fig, B(r,3));
end

save_final_png(fig, 'encounter_plane');
nFrames = anim.n_frames;
animation_close(anim);

R.stage_cells = cells - 1;
R.stage_xi = xiS;
R.stage_pc = pcS;
R.n_frames = nFrames;

if ~isfolder(fullfile(root, 'results'))
    mkdir(fullfile(root, 'results'));
end
fid = fopen(fullfile(root, 'results', 'viz_encounter_plane.json'), 'w');
fprintf(fid, '%s', jsonencode(R));
fclose(fid);

close(fig);
out = R;
end
