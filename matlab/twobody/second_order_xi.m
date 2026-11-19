function xi2 = second_order_xi(S, u1, u2)
%SECOND_ORDER_XI Linear burn-to-miss map plus its second-order correction in the burn sizes.
delta = hypot(S.a + S.miss_zeta, S.miss_xi) - S.a;
xi2 = S.miss_xi + S.gain * (u1 + u2) + (7 * S.gain / (2 * S.v)) * (u1.^2 - u2.^2) ...
    + (6 * delta / S.a) * S.gain * u2;
end
