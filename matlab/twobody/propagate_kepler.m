function [r, v] = propagate_kepler(r0, v0, t, mu)
if t == 0
    r = r0;
    v = v0;
    return
end
sm = sqrt(mu);
r0n = norm(r0);
vr0 = dot(r0, v0) / r0n;
alpha = 2 / r0n - dot(v0, v0) / mu;
if alpha > 0
    chi = sm * alpha * t;
else
    chi = sm * t / r0n;
end
for it = 1:50
    z = alpha * chi^2;
    [C, Sz] = stumpff(z);
    F = r0n * vr0 / sm * chi^2 * C + (1 - alpha * r0n) * chi^3 * Sz + r0n * chi - sm * t;
    dF = r0n * vr0 / sm * chi * (1 - z * Sz) + (1 - alpha * r0n) * chi^2 * C + r0n;
    step = F / dF;
    chi = chi - step;
    if abs(step) <= 1e-14 * max(abs(chi), eps)
        break
    end
end
z = alpha * chi^2;
[C, Sz] = stumpff(z);
f = 1 - chi^2 / r0n * C;
g = t - chi^3 / sm * Sz;
r = f * r0 + g * v0;
rn = norm(r);
fdot = sm / (rn * r0n) * (z * Sz - 1) * chi;
gdot = 1 - chi^2 / rn * C;
v = fdot * r0 + gdot * v0;
end

function [C, S] = stumpff(z)
if abs(z) < 1e-3
    C = 1/2 - z/24 + z^2/720 - z^3/40320 + z^4/3628800;
    S = 1/6 - z/120 + z^2/5040 - z^3/362880 + z^4/39916800;
elseif z > 0
    q = sqrt(z);
    C = (1 - cos(q)) / z;
    S = (q - sin(q)) / q^3;
else
    q = sqrt(-z);
    C = (cosh(q) - 1) / (-z);
    S = (sinh(q) - q) / q^3;
end
end
