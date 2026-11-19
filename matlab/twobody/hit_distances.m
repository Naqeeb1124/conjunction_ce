function d = hit_distances(S, out, x, z, s, tol)
%HIT_DISTANCES Closest-approach distance for each sampled position of satellite 2.
%   Satellite 1 keeps its post-burn state; satellite 2 is placed at r1 + x xi + z zeta + s s_hat
%   (encounter axes of out) with its post-burn velocity, and both are propagated with Kepler.
n = numel(x);
d = zeros(n, 1);
for k = 1:n
    r2k = out.r1 + x(k) * out.B.xi + z(k) * out.B.zeta + s(k) * out.B.s;
    ca = closest_approach(out.r1, out.v1, r2k, out.v2, S.mu, 'kepler', tol);
    d(k) = norm(ca.rho);
end
end
