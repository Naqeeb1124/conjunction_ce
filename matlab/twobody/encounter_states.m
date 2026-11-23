function [r1, v1, r2, v2, B] = encounter_states(S)
half = S.phi / 2;
t1 = [0 cos(half) sin(half)];
t2 = [0 cos(half) -sin(half)];
r1 = [S.a 0 0];
r2 = r1 + [S.miss_zeta S.miss_xi 0];
v1 = S.v * t1;
v2 = S.v * t2;
B = encounter_basis(v1, v2);
end
