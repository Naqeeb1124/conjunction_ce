function mc = pc_montecarlo_naive(S, out, n, seed)
%PC_MONTECARLO_NAIVE Plain Monte Carlo collision probability, for comparison only.
%   Positions are drawn from the Gaussian miss distribution; hits are counted directly and
%   an exact Clopper-Pearson 95 percent interval is given.
P = load_params();
tb = P.two_body_check;
rng(seed, 'twister');
x = out.xi + S.sigma_xi * randn(n, 1);
z = out.zeta + S.sigma_zeta * randn(n, 1);
s = tb.sigma_s * randn(n, 1);
d = hit_distances(S, out, x, z, s, tb.tca_tol_s);
hits = sum(d < S.hbr);
if hits == 0
    lo = 0;
else
    lo = betaincinv(0.025, hits, n - hits + 1);
end
if hits == n
    hi = 1;
else
    hi = betaincinv(0.975, hits + 1, n - hits);
end
mc.hits = hits;
mc.n = n;
mc.pc = hits / n;
mc.ci95 = [lo hi];
mc.pc_foster = pc2d(out.xi, out.zeta, S.sigma_xi, S.sigma_zeta, S.hbr);
mc.expected_hits = n * mc.pc_foster;
end
