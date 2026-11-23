function pc = pc2d(xm, ym, sx, sy, R)
persistent u wrow
if isempty(u)
    k = 1:127;
    beta = k ./ sqrt(4 * k.^2 - 1);
    J = diag(beta, 1) + diag(beta, -1);
    [V, D] = eig(J);
    u = diag(D);
    wrow = (2 * (V(1, :).').^2).';
end
Phi = @(z) 0.5 * erfc(-z / sqrt(2));

z = xm + ym + sx + sy + R;
outSize = size(z);
N = numel(z);
col = @(v) v(:) + zeros(N, 1);
xm = col(xm); ym = col(ym); sx = col(sx); sy = col(sy); R = col(R);

swap = sy > sx;
am = xm;
am(swap) = ym(swap);
bm = abs(ym);
bm(swap) = abs(xm(swap));
as = max(sx, sy);
bs = min(sx, sy);

t = (pi / 2) * u.';
a = R .* sin(t);
h = R .* cos(t);
dens = exp(-0.5 * ((a - am) ./ as).^2) ./ (sqrt(2 * pi) * as);
inner = Phi((h - bm) ./ bs) - Phi((-h - bm) ./ bs);
pcv = (pi / 2) * sum(wrow .* dens .* inner .* h, 2);
pc = reshape(pcv, outSize);
end
