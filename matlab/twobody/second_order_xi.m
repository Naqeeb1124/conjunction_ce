function xi2 = second_order_xi(S, u1, u2)
delta = hypot(S.a + S.miss_zeta, S.miss_xi) - S.a;
xi2 = S.miss_xi + S.gain * (u1 + u2) + (7 * S.gain / (2 * S.v)) * (u1.^2 - u2.^2) ...
    + (6 * delta / S.a) * S.gain * u2;
end
