function [xi, gain] = miss_shift(S, u1, u2)
%MISS_SHIFT Along-track miss after the two burns: xi = miss_xi + gain (u1 + u2).
%   [xi, gain] = miss_shift(S, u1, u2) with burn sizes u1, u2 in m/s (scalars or arrays).
gain = S.gain;
xi = S.miss_xi + gain .* (u1 + u2);
end
