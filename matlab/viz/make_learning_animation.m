function out = make_learning_animation()
% Renders the learning animation (MP4, GIF, final PNG) and writes results/viz_learning.json.
root = setup_paths();
E = jsondecode(fileread(fullfile(root, 'expected.json')));
L = E.learning;
st = viz_style();

kvec = L.k(:);
pubY = 100 * L.rules.public.coordinator_cost(:);
m0 = 100 * L.rules.private_m0.coordinator_cost(:);
uni = 100 * L.rules.private_uniform.coordinator_cost(:);
tgt = 100 * L.best_targeted.coordinator_cost(:);
known = 100 * L.known_rule_cost.private_m0;
kx = L.crossover_k.targeted;

titleText = 'Private recommendations pay off once operators have seen enough events';
subtitleText = ['Realised coordinator cost when both operators learned the rule from k past events ' ...
    '(costs known). Lower is better.'];
[fig, ax] = viz_figure(titleText, subtitleText);
set(ax, 'XScale', 'log', 'YScale', 'log', 'XLim', [1 1e4], 'YLim', [2 130], ...
    'Position', [0.075 0.13 0.60 0.645], ...
    'XTick', [1 10 100 1000 10000], 'XTickLabel', {'1', '10', '100', '1,000', '10,000'}, ...
    'YTick', [3 10 30 100], 'YTickLabel', {'3', '10', '30', '100'});
xlabel(ax, 'k: past events each operator has seen');
ylabel(ax, 'realised coordinator cost (cm/s equivalent)');

hPub = plot(ax, kvec, pubY, '-', 'Color', st.ink2, 'LineWidth', 2.4);
hKnown = plot(ax, [1 1e4], [known known], '-', 'Color', st.axis, 'LineWidth', 1.4);
text(ax, 1.15, known * 0.97, sprintf('limit if the rule were known: %.2f', known), ...
    'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.muted, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'Interpreter', 'none');

hM0 = plot(ax, NaN, NaN, '-', 'Color', st.orange, 'LineWidth', st.lw);
hUni = plot(ax, NaN, NaN, '-', 'Color', st.aqua, 'LineWidth', st.lw);
hTgt = plot(ax, NaN, NaN, '-', 'Color', st.blue, 'LineWidth', st.lw);
hM0h = plot(ax, NaN, NaN, 'o', 'MarkerSize', 9, 'MarkerFaceColor', st.orange, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');
hUnih = plot(ax, NaN, NaN, 'o', 'MarkerSize', 9, 'MarkerFaceColor', st.aqua, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');
hTgth = plot(ax, NaN, NaN, 'o', 'MarkerSize', 9, 'MarkerFaceColor', st.blue, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');
hCross = plot(ax, NaN, NaN, 'o', 'MarkerSize', 14, 'MarkerFaceColor', st.blue, ...
    'MarkerEdgeColor', 'w', 'LineStyle', 'none');
hNote = text(ax, 110, 4.2, sprintf('targeted private rule beats the\npublic rule from k = %d', kx), ...
    'FontName', st.font, 'FontSize', st.fs_note, 'Color', st.ink, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
    'Interpreter', 'none', 'Visible', 'off');
hK = text(ax, 1.2, 125, '', 'FontName', st.font, 'FontSize', st.fs_label, 'Color', st.ink, ...
    'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'Interpreter', 'none');

lg = legend(ax, [hPub, hM0, hUni, hTgt], ...
    {'fixed / public rule (nothing to learn)', 'private, no margin', ...
     'private, same margin on every constraint', 'private, targeted margin chosen for each k'}, ...
    'Location', 'eastoutside', 'Box', 'off', 'FontName', st.font, 'FontSize', st.fs_note);
set(lg, 'Color', 'w', 'Units', 'normalized', 'ItemTokenSize', [12 12]);
set(ax, 'Position', [0.075 0.13 0.60 0.645]);
lgPos = get(lg, 'Position');
set(lg, 'Position', [0.680 lgPos(2) lgPos(3) lgPos(4)]);

kcs = 10 .^ linspace(0, 4, 150);
kcs(end) = 1e4;

lines = {hM0, hUni, hTgt};
heads = {hM0h, hUnih, hTgth};
vals = {m0, uni, tgt};

anim = animation_open('learning', 30);
for i = 1:numel(kcs)
    kc = kcs(i);
    for s = 1:3
        yv = vals{s};
        j = find(kvec <= kc);
        xs = kvec(j);
        ys = yv(j);
        if kc > xs(end) && j(end) < numel(kvec)
            a = j(end);
            b = a + 1;
            t = (log10(kc) - log10(kvec(a))) / (log10(kvec(b)) - log10(kvec(a)));
            ys(end+1) = 10 ^ (log10(yv(a)) + t * (log10(yv(b)) - log10(yv(a)))); %#ok<AGROW>
            xs(end+1) = kc; %#ok<AGROW>
        end
        set(lines{s}, 'XData', xs, 'YData', ys);
        set(heads{s}, 'XData', xs(end), 'YData', ys(end));
    end
    set(hK, 'String', sprintf('k = %d past events', round(kc)));
    if kc >= kx
        ix = find(kvec == kx, 1);
        set(hCross, 'XData', kvec(ix), 'YData', tgt(ix));
        set(hNote, 'Visible', 'on');
    end
    rep = 1;
    if i == numel(kcs)
        rep = 60;
    end
    anim = animation_frame(anim, fig, rep);
end

save_final_png(fig, 'learning');
info = animation_close(anim); %#ok<NASGU>

R.k = get(hM0, 'XData');
R.k = R.k(:);
R.public = get(hPub, 'YData');
R.public = R.public(:);
R.private_m0 = get(hM0, 'YData');
R.private_m0 = R.private_m0(:);
R.private_uniform = get(hUni, 'YData');
R.private_uniform = R.private_uniform(:);
R.best_targeted = get(hTgt, 'YData');
R.best_targeted = R.best_targeted(:);

if ~isfolder(fullfile(root, 'results'))
    mkdir(fullfile(root, 'results'));
end
fid = fopen(fullfile(root, 'results', 'viz_learning.json'), 'w');
fprintf(fid, '%s', jsonencode(R));
fclose(fid);

close(fig);
out = R;
end
