function [p, q, p_dual] = mixed_nash(c1, c2)
qN = (c1(2, 2) - c1(1, 2)) / (c1(1, 1) - c1(1, 2) - c1(2, 1) + c1(2, 2));
pN = (c2(2, 2) - c2(2, 1)) / (c2(1, 1) - c2(2, 1) - c2(1, 2) + c2(2, 2));
p = [pN; 1 - pN; 0];
q = [qN; 1 - qN; 0];
p_dual = pN * qN;
end
