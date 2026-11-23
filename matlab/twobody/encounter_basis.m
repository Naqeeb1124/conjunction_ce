function B = encounter_basis(v1, v2)
t1 = v1 / norm(v1);
t2 = v2 / norm(v2);
s = (v2 - v1) / norm(v2 - v1);
xi = (t1 + t2) / norm(t1 + t2);
xi = xi - dot(xi, s) * s;
xi = xi / norm(xi);
zeta = cross(s, xi);
B = struct('xi', xi, 'zeta', zeta, 's', s);
end
