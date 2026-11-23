function [xi, gain] = miss_shift(S, u1, u2)
gain = S.gain;
xi = S.miss_xi + gain .* (u1 + u2);
end
