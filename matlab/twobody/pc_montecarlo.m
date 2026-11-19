function mc = pc_montecarlo(S, out, n, seed)
%PC_MONTECARLO Importance-sampled collision probability of a propagated encounter.
%   Relative positions are drawn uniformly on a disc of radius mc_proposal_radius_hbr*hbr in the
%   encounter plane (plus a Gaussian offset along the relative velocity), each sample is
%   propagated to its true closest approach, and hits are weighted by the Gaussian miss density
%   divided by the uniform proposal density.
P = load_params();
tb = P.two_body_check;
R = S.hbr;
rq = tb.mc_proposal_radius_hbr * R;
rng(seed, 'twister');
rad = rq * sqrt(rand(n, 1));
ang = 2 * pi * rand(n, 1);
x = rad .* cos(ang);
z = rad .* sin(ang);
s = tb.sigma_s * randn(n, 1);
d = hit_distances(S, out, x, z, s, tb.tca_tol_s);
hit = d < R;
wgt = pi * rq^2 * exp(-0.5 * ((x - out.xi) / S.sigma_xi).^2 - 0.5 * ((z - out.zeta) / S.sigma_zeta).^2) ...
    / (2 * pi * S.sigma_xi * S.sigma_zeta);
hw = hit .* wgt;
mc.pc = mean(hw);
mc.se = std(hw) / sqrt(n);
mc.hit_fraction = mean(hit);
mc.plane_agreement = mean(hit == (hypot(x, z) < R));
mc.pc_foster = pc2d(out.xi, out.zeta, S.sigma_xi, S.sigma_zeta, R);
mc.z = (mc.pc - mc.pc_foster) / mc.se;
mc.n = n;
mc.seed = seed;
end
