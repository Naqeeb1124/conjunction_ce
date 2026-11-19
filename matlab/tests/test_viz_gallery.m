function test_viz_gallery()
% Checks the gallery animation files and the per-case recommendations and table values against the reference.
root = setup_paths();
check_animation_files('gallery');

R = jsondecode(fileread(fullfile(root, 'results', 'viz_gallery.json')));
E = load_expected();
C = R.cases;
n = numel(E.gallery);

a1 = zeros(n, 1);
a2 = zeros(n, 1);
prob = zeros(n, 1);
pcb = zeros(n, 1);
pca = zeros(n, 1);
rec = zeros(n, 2);
refProb = zeros(n, 1);
refPcb = zeros(n, 1);
refPca = zeros(n, 1);
xiRef = zeros(n, 1);
for i = 1:n
    g = E.gallery(i);
    a1(i) = C(i).a1;
    a2(i) = C(i).a2;
    prob(i) = C(i).prob;
    pcb(i) = C(i).pc_before;
    pca(i) = C(i).pc_after;
    rec(i, :) = reshape(g.ce_modal_recommendation, 1, 2);
    refProb(i) = g.ce_modal_probability;
    refPcb(i) = g.pc_table(1, 1);
    refPca(i) = g.pc_table(a1(i) + 1, a2(i) + 1);
    xiRef(i) = g.miss_xi_table(a1(i) + 1, a2(i) + 1);
end

tieFlag = logical([C.tie])';
eqRow = (a1 == rec(:, 1)) & (a2 == rec(:, 2));
swRow = tieFlag & (rec(:, 1) == a2) & (rec(:, 2) == a1);
nEqual = nnz(eqRow);
nTie = nnz(~eqRow & swRow);
tieIds = find(~eqRow & swRow)';
check_true('gallery recommendation: equal to the reference, or an exact tie of the linear program', ...
    all(eqRow | swRow), sprintf('%d equal to the reference; %d differ and are exact ties (cases %s)', nEqual, nTie, num2str(tieIds)));

% Reference tie: at least two pure Nash equilibria whose coordinator cost equals the CE objective.
refTie = false(n, 1);
for i = 1:n
    g = E.gallery(i);
    pn = g.pure_nash;
    if isempty(pn)
        pairs = zeros(0, 2);
    elseif numel(pn) == 2
        pairs = reshape(pn, 1, 2);
    else
        pairs = pn;
    end
    nOpt = 0;
    for k = 1:size(pairs, 1)
        c = g.coordinator_cost_table(pairs(k, 1) + 1, pairs(k, 2) + 1);
        if abs(c - g.ce_objective) <= 1e-12 * abs(g.ce_objective)
            nOpt = nOpt + 1;
        end
    end
    refTie(i) = nOpt >= 2;
end
check_true('gallery tie flag agrees with the reference (two optimal pure rules of equal cost)', ...
    isequal(tieFlag, refTie), sprintf('%d of %d cases are ties (cases %s)', nnz(refTie), n, num2str(find(refTie)')));
check_close('gallery modal probability', prob, refProb, 0, 1e-6);
check_close('gallery pc_before', pcb, refPcb, 1e-3, 0);
check_close('gallery xi_after', [C.xi_after]', xiRef, 1e-6, 0);
check_close('gallery pc_after', pca, refPca, 1e-3, 0);
end
