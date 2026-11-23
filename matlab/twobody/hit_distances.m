function d = hit_distances(S, out, x, z, s, tol)
n = numel(x);
d = zeros(n, 1);
for k = 1:n
    r2k = out.r1 + x(k) * out.B.xi + z(k) * out.B.zeta + s(k) * out.B.s;
    ca = closest_approach(out.r1, out.v1, r2k, out.v2, S.mu, 'kepler', tol);
    d(k) = norm(ca.rho);
end
end
