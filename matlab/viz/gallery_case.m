function G = gallery_case(P, i)
Si = load_scenario(P.gallery(i).scenario, P.constants);
Ti = cost_matrices(Si);
[x, ~] = solve_ce(Ti.c1, Ti.c2, Ti.obj);
xf = reshape(x.', [], 1);
[prob, k] = max(xf);
a1 = floor((k - 1) / 3) + 1;
a2 = mod(k - 1, 3) + 1;

u1 = Si.burns(a1);
u2 = Si.burns(a2);
G.id = P.gallery(i).id;
G.S = Si;
G.a1 = a1;
G.a2 = a2;
G.prob = prob;
G.d1 = 3 * u1 * Si.T_lead;
G.d2 = 3 * u2 * Si.T_lead;
G.half = Si.phi / 2;
G.xi0 = Si.miss_xi;
G.xi_after = Ti.xi(a1, a2);
G.pc_before = Ti.pc(1, 1);
G.pc_after = Ti.pc(a1, a2);
G.miss_before = hypot(G.xi0, Si.miss_zeta);
G.miss_after = hypot(G.xi_after, Si.miss_zeta);
G.sig_s = P.gallery(i).sigma_s;
G.tau_star = (G.d1 - G.d2) / (2 * Si.v);
G.nobody = (a1 == 1) && (a2 == 1);

G.tie = false;
if prob >= 1 - 1e-9 && a1 ~= a2
    b1 = a2;
    b2 = a1;
    okc1 = all(Ti.c1(b1, b2) <= Ti.c1(1:3, b2) + 1e-15);
    okc2 = all(Ti.c2(b1, b2) <= Ti.c2(b1, 1:3) + 1e-15);
    oko = abs(Ti.obj(b1, b2) - Ti.obj(a1, a2)) <= 1e-12 * abs(Ti.obj(a1, a2));
    G.tie = okc1 && okc2 && oko;
end
end
