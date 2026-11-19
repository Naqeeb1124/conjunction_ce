function S = load_scenario(raw, constants)
%LOAD_SCENARIO Build the scenario struct with the derived orbit and cost quantities.
%   S = load_scenario() uses the baseline scenario and constants from params.json.
%   S = load_scenario(raw, constants) copies the fields of raw and adds mu, r_earth, burn_sign,
%   phi (rad), a (m), n (rad/s), T (s), v (m/s), T_lead (s), gain (m per m/s), and
%   fuel_price (USD per m/s). Vector fields are stored as columns.
if nargin == 0
    P = load_params();
    raw = P.scenario;
    constants = P.constants;
end
S = struct();
names = fieldnames(raw);
for i = 1:numel(names)
    v = raw.(names{i});
    if isnumeric(v) && isvector(v) && numel(v) > 1
        v = v(:);
    end
    S.(names{i}) = v;
end
S.mu = constants.mu_earth;
S.r_earth = constants.r_earth;
S.burn_sign = constants.burn_sign(:);
S.phi = deg2rad(S.crossing_angle_deg);
% Circular orbit radius, mean motion n = sqrt(mu/a^3), period T, circular speed v = sqrt(mu/a)
S.a = S.r_earth + S.altitude;
S.n = sqrt(S.mu / S.a^3);
S.T = 2 * pi / S.n;
S.v = sqrt(S.mu / S.a);
S.T_lead = S.lead_orbits * S.T;
% Along-track miss change per unit burn: 3 T_lead cos(phi/2) (Clohessy-Wiltshire-type factor)
S.gain = 3 * S.T_lead * cos(S.phi / 2);
S.fuel_price = S.value ./ S.dv_budget;
S.fuel_price = S.fuel_price(:);
end
