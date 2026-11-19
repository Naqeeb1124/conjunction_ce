function mdl = build_conjunction_sim()
%BUILD_CONJUNCTION_SIM Create simulink/conjunction_sim.slx for one conjunction event.
%   mdl = build_conjunction_sim() builds the model from nothing and returns its name.
%   Signal flow: Coordinator -> (a1, a2) -> Burn -> Satellite 1 / Satellite 2 -> Miss distance.
%   Every number is a workspace variable supplied at run time: x_rule_flat, u_draw, burns,
%   burn_sign, mu_earth, r1_0, v1_0, r2_0, v2_0 (pre-burn states), T_lead, ode_reltol, ode_abstol.
%   Coordinator: inverse CDF of the rule, k = 1 + #{j : cumsum(x)_j <= u}, clipped to 9;
%     a1 = floor((k-1)/3) + 1, a2 = mod(k-1, 3) + 1.
%   Burn: v_after = v_before + sign_i * u_i * v_before / |v_before|, u_i = burns(a_i).
%   Satellite: r' = v, v' = -mu_earth r / |r|^3, integrators started from the external ports
%     (position from the pre-burn position, velocity from the post-burn velocity).
%   Miss distance: rho = r2 - r1, w = v2 - v1, t_corr = -(rho.w)/(w.w), m = rho + w t_corr.
%   The encounter-plane projection of m is done by the caller with encounter_basis.
root = setup_paths();
mdl = 'conjunction_sim';
folder = fullfile(root, 'simulink');
if bdIsLoaded(mdl)
    close_system(mdl, 0);
end
slx = fullfile(folder, [mdl '.slx']);
if exist(slx, 'file')
    delete(slx);
end
new_system(mdl);

% Short forms for adding blocks and lines.
blk = @(sys, lib, name, varargin) add_block(['simulink/' lib], [sys '/' name], varargin{:});
lnk = @(sys, a, b) add_line(sys, a, b, 'autorouting', 'on');
inport = @(sys, name, n) add_block('simulink/Sources/In1', [sys '/' name], 'Port', num2str(n));
outport = @(sys, name, n) add_block('simulink/Sinks/Out1', [sys '/' name], 'Port', num2str(n));
tows = @(sys, name, var) add_block('simulink/Sinks/To Workspace', [sys '/' name], ...
    'VariableName', var, 'SaveFormat', 'Array', 'MaxDataPoints', 'inf', 'SampleTime', '-1');

top = mdl;

% Empty subsystems, one per stage.
for name = {'Coordinator', 'Burn', 'Satellite 1', 'Satellite 2', 'Miss distance'}
    sys = [top '/' name{1}];
    add_block('built-in/Subsystem', sys);
    delete_line(find_system(sys, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line'));
    delete_block(find_system(sys, 'SearchDepth', 1, 'BlockType', 'Inport'));
    delete_block(find_system(sys, 'SearchDepth', 1, 'BlockType', 'Outport'));
end

%% Coordinator
C = [top '/Coordinator'];
blk(C, 'Sources/Constant', 'x_rule', 'Value', 'x_rule_flat(:)');
blk(C, 'Sources/Constant', 'lower_triangle', 'Value', 'tril(ones(numel(x_rule_flat)))');
blk(C, 'Math Operations/Product', 'cumulative_sum', 'Inputs', '2', 'Multiplication', 'Matrix(*)');
blk(C, 'Sources/Constant', 'draw', 'Value', 'u_draw');
blk(C, 'Logic and Bit Operations/Relational Operator', 'cdf_le_draw', 'Operator', '<=');
blk(C, 'Signal Attributes/Data Type Conversion', 'to_double', 'OutDataTypeStr', 'double');
blk(C, 'Math Operations/Sum of Elements', 'count', 'Inputs', '+');
blk(C, 'Math Operations/Bias', 'plus_one', 'Bias', '1');
blk(C, 'Discontinuities/Saturation', 'clip_k', 'UpperLimit', 'numel(x_rule_flat)', 'LowerLimit', '1');
blk(C, 'Math Operations/Bias', 'k_minus_one', 'Bias', '-1');
blk(C, 'Sources/Constant', 'n_actions', 'Value', '3');
blk(C, 'Math Operations/Product', 'row_div', 'Inputs', '*/');
blk(C, 'Math Operations/Rounding Function', 'row_floor', 'Operator', 'floor');
blk(C, 'Math Operations/Bias', 'a1_one_based', 'Bias', '1');
blk(C, 'Math Operations/Math Function', 'col_mod', 'Operator', 'mod');
blk(C, 'Math Operations/Bias', 'a2_one_based', 'Bias', '1');
outport(C, 'a1', 1);
outport(C, 'a2', 2);
lnk(C, 'lower_triangle/1', 'cumulative_sum/1');
lnk(C, 'x_rule/1', 'cumulative_sum/2');
lnk(C, 'cumulative_sum/1', 'cdf_le_draw/1');
lnk(C, 'draw/1', 'cdf_le_draw/2');
lnk(C, 'cdf_le_draw/1', 'to_double/1');
lnk(C, 'to_double/1', 'count/1');
lnk(C, 'count/1', 'plus_one/1');
lnk(C, 'plus_one/1', 'clip_k/1');
lnk(C, 'clip_k/1', 'k_minus_one/1');
lnk(C, 'k_minus_one/1', 'row_div/1');
lnk(C, 'n_actions/1', 'row_div/2');
lnk(C, 'row_div/1', 'row_floor/1');
lnk(C, 'row_floor/1', 'a1_one_based/1');
lnk(C, 'a1_one_based/1', 'a1/1');
lnk(C, 'k_minus_one/1', 'col_mod/1');
lnk(C, 'n_actions/1', 'col_mod/2');
lnk(C, 'col_mod/1', 'a2_one_based/1');
lnk(C, 'a2_one_based/1', 'a2/1');

%% Burn
B = [top '/Burn'];
inport(B, 'a1', 1);
inport(B, 'a2', 2);
blk(B, 'Sources/Constant', 'burn_table', 'Value', 'burns(:)');
for i = 1:2
    s = num2str(i);
    blk(B, 'Sources/Constant', ['v' s '_pre'], 'Value', ['v' s '_0(:)']);
    blk(B, 'Sources/Constant', ['r' s '_pre'], 'Value', ['r' s '_0(:)']);
    blk(B, 'Signal Routing/Selector', ['u' s], 'IndexMode', 'One-based', ...
        'NumberOfDimensions', '1', 'IndexOptionArray', {'Index vector (port)'}, 'InputPortWidth', '-1');
    blk(B, 'Sources/Constant', ['sign' s], 'Value', ['burn_sign(' s ')']);
    blk(B, 'Math Operations/Product', ['signed_u' s], 'Inputs', '**');
    blk(B, 'Math Operations/Dot Product', ['vv' s]);
    blk(B, 'Math Operations/Sqrt', ['speed' s]);
    blk(B, 'Math Operations/Product', ['direction' s], 'Inputs', '*/');
    blk(B, 'Math Operations/Product', ['dv' s], 'Inputs', '**');
    blk(B, 'Math Operations/Sum', ['v' s '_post'], 'Inputs', '++');
    outport(B, ['v' s '_post_out'], i);
    outport(B, ['r' s '_pre_out'], 2 + i);
    outport(B, ['u' s '_out'], 4 + i);
    lnk(B, ['burn_table/1'], ['u' s '/1']);
    lnk(B, ['a' s '/1'], ['u' s '/2']);
    lnk(B, ['u' s '/1'], ['signed_u' s '/1']);
    lnk(B, ['sign' s '/1'], ['signed_u' s '/2']);
    lnk(B, ['v' s '_pre/1'], ['vv' s '/1']);
    lnk(B, ['v' s '_pre/1'], ['vv' s '/2']);
    lnk(B, ['vv' s '/1'], ['speed' s '/1']);
    lnk(B, ['v' s '_pre/1'], ['direction' s '/1']);
    lnk(B, ['speed' s '/1'], ['direction' s '/2']);
    lnk(B, ['direction' s '/1'], ['dv' s '/1']);
    lnk(B, ['signed_u' s '/1'], ['dv' s '/2']);
    lnk(B, ['v' s '_pre/1'], ['v' s '_post/1']);
    lnk(B, ['dv' s '/1'], ['v' s '_post/2']);
    lnk(B, ['v' s '_post/1'], ['v' s '_post_out/1']);
    lnk(B, ['r' s '_pre/1'], ['r' s '_pre_out/1']);
    lnk(B, ['u' s '/1'], ['u' s '_out/1']);
end

%% Satellites: two-body dynamics
for i = 1:2
    s = num2str(i);
    Y = [top '/Satellite ' s];
        inport(Y, 'r_pre', 1);
    inport(Y, 'v_post', 2);
    blk(Y, 'Continuous/Integrator', 'velocity', 'InitialConditionSource', 'external');
    blk(Y, 'Continuous/Integrator', 'position', 'InitialConditionSource', 'external');
    blk(Y, 'Math Operations/Dot Product', 'r_dot_r');
    blk(Y, 'Sources/Constant', 'exponent', 'Value', '-3/2');
    blk(Y, 'Math Operations/Math Function', 'inverse_cube', 'Operator', 'pow');
    blk(Y, 'Math Operations/Product', 'r_scaled', 'Inputs', '**');
    blk(Y, 'Math Operations/Gain', 'minus_mu', 'Gain', '-mu_earth');
    outport(Y, 'r', 1);
    outport(Y, 'v', 2);
    lnk(Y, 'r_pre/1', 'position/2');
    lnk(Y, 'v_post/1', 'velocity/2');
    lnk(Y, 'velocity/1', 'position/1');
    lnk(Y, 'position/1', 'r_dot_r/1');
    lnk(Y, 'position/1', 'r_dot_r/2');
    lnk(Y, 'r_dot_r/1', 'inverse_cube/1');
    lnk(Y, 'exponent/1', 'inverse_cube/2');
    lnk(Y, 'position/1', 'r_scaled/1');
    lnk(Y, 'inverse_cube/1', 'r_scaled/2');
    lnk(Y, 'r_scaled/1', 'minus_mu/1');
    lnk(Y, 'minus_mu/1', 'velocity/1');
    lnk(Y, 'position/1', 'r/1');
    lnk(Y, 'velocity/1', 'v/1');
end

%% Miss distance
M = [top '/Miss distance'];
inport(M, 'r1', 1);
inport(M, 'v1', 2);
inport(M, 'r2', 3);
inport(M, 'v2', 4);
blk(M, 'Math Operations/Sum', 'rho', 'Inputs', '+-');
blk(M, 'Math Operations/Sum', 'w', 'Inputs', '+-');
blk(M, 'Math Operations/Dot Product', 'rho_rho');
blk(M, 'Math Operations/Sqrt', 'range');
blk(M, 'Math Operations/Dot Product', 'rho_w');
blk(M, 'Math Operations/Dot Product', 'w_w');
blk(M, 'Math Operations/Product', 'ratio', 'Inputs', '*/');
blk(M, 'Math Operations/Gain', 'negate', 'Gain', '-1');
blk(M, 'Math Operations/Product', 'w_shift', 'Inputs', '**');
blk(M, 'Math Operations/Sum', 'm_vec', 'Inputs', '++');
blk(M, 'Math Operations/Dot Product', 'm_m');
blk(M, 'Math Operations/Sqrt', 'miss');
outport(M, 'miss_out', 1);
tows(M, 'log_range', 'range_log');
tows(M, 'log_tcorr', 'tcorr_log');
tows(M, 'log_m', 'm_log');
tows(M, 'log_v1', 'v1_log');
tows(M, 'log_v2', 'v2_log');
lnk(M, 'r2/1', 'rho/1');
lnk(M, 'r1/1', 'rho/2');
lnk(M, 'v2/1', 'w/1');
lnk(M, 'v1/1', 'w/2');
lnk(M, 'rho/1', 'rho_rho/1');
lnk(M, 'rho/1', 'rho_rho/2');
lnk(M, 'rho_rho/1', 'range/1');
lnk(M, 'rho/1', 'rho_w/1');
lnk(M, 'w/1', 'rho_w/2');
lnk(M, 'w/1', 'w_w/1');
lnk(M, 'w/1', 'w_w/2');
lnk(M, 'rho_w/1', 'ratio/1');
lnk(M, 'w_w/1', 'ratio/2');
lnk(M, 'ratio/1', 'negate/1');
lnk(M, 'w/1', 'w_shift/1');
lnk(M, 'negate/1', 'w_shift/2');
lnk(M, 'rho/1', 'm_vec/1');
lnk(M, 'w_shift/1', 'm_vec/2');
lnk(M, 'm_vec/1', 'm_m/1');
lnk(M, 'm_vec/1', 'm_m/2');
lnk(M, 'm_m/1', 'miss/1');
lnk(M, 'miss/1', 'miss_out/1');
lnk(M, 'range/1', 'log_range/1');
lnk(M, 'negate/1', 'log_tcorr/1');
lnk(M, 'm_vec/1', 'log_m/1');
lnk(M, 'v1/1', 'log_v1/1');
lnk(M, 'v2/1', 'log_v2/1');

%% Top level wiring and logging
tows(top, 'log_a1', 'a1_log');
tows(top, 'log_a2', 'a2_log');
tows(top, 'log_u1', 'u1_log');
tows(top, 'log_u2', 'u2_log');
tows(top, 'log_miss', 'miss_log');
lnk(top, 'Coordinator/1', 'Burn/1');
lnk(top, 'Coordinator/2', 'Burn/2');
lnk(top, 'Coordinator/1', 'log_a1/1');
lnk(top, 'Coordinator/2', 'log_a2/1');
lnk(top, 'Burn/3', 'Satellite 1/1');
lnk(top, 'Burn/1', 'Satellite 1/2');
lnk(top, 'Burn/4', 'Satellite 2/1');
lnk(top, 'Burn/2', 'Satellite 2/2');
lnk(top, 'Burn/5', 'log_u1/1');
lnk(top, 'Burn/6', 'log_u2/1');
lnk(top, 'Satellite 1/1', 'Miss distance/1');
lnk(top, 'Satellite 1/2', 'Miss distance/2');
lnk(top, 'Satellite 2/1', 'Miss distance/3');
lnk(top, 'Satellite 2/2', 'Miss distance/4');
lnk(top, 'Miss distance/1', 'log_miss/1');

%% Solver: all values are variables
set_param(mdl, 'SolverType', 'Variable-step', 'Solver', 'ode113', ...
    'StartTime', '0', 'StopTime', 'T_lead', 'RelTol', 'ode_reltol', 'AbsTol', 'ode_abstol', ...
    'SaveTime', 'on', 'TimeSaveName', 'tout', 'SaveOutput', 'off', 'SaveState', 'off', ...
    'ReturnWorkspaceOutputs', 'on', 'SignalLogging', 'off');

%% Layout, left to right
for name = {'Coordinator', 'Burn', 'Satellite 1', 'Satellite 2', 'Miss distance'}
    Simulink.BlockDiagram.arrangeSystem([top '/' name{1}]);
end
Simulink.BlockDiagram.arrangeSystem(top);

save_system(mdl, slx);
close_system(mdl, 0);
end
