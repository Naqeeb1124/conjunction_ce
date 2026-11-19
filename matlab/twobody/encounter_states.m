function [r1, v1, r2, v2, B] = encounter_states(S)
%ENCOUNTER_STATES Nominal states of both satellites at the closest-approach time (t = 0).
%   Satellite 1 sits at [a 0 0]; satellite 2 is offset by the predicted miss [miss_zeta miss_xi 0].
%   The velocities are symmetric about the y axis with crossing angle phi. Row vectors, SI units.
half = S.phi / 2;
t1 = [0 cos(half) sin(half)];
t2 = [0 cos(half) -sin(half)];
r1 = [S.a 0 0];
r2 = r1 + [S.miss_zeta S.miss_xi 0];
v1 = S.v * t1;
v2 = S.v * t2;
B = encounter_basis(v1, v2);
end
