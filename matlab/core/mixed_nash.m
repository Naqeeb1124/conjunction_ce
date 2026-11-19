function [p, q, p_dual] = mixed_nash(c1, c2)
%MIXED_NASH Mixed equilibrium over the no-burn and small-burn actions only.
%   [p, q, p_dual] = mixed_nash(c1, c2): p is operator 1's mix, q is operator 2's mix, each
%   ordered [no burn; small burn; large burn] with zero weight on the large burn.
%   p_dual is the probability that both operators do not burn under independent mixing.
qN = (c1(2, 2) - c1(1, 2)) / (c1(1, 1) - c1(1, 2) - c1(2, 1) + c1(2, 2));
pN = (c2(2, 2) - c2(2, 1)) / (c2(1, 1) - c2(2, 1) - c2(1, 2) + c2(2, 2));
p = [pN; 1 - pN; 0];
q = [qN; 1 - qN; 0];
p_dual = pN * qN;
end
