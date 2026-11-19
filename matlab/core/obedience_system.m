function [A, Pm, info] = obedience_system(c1, c2)
%OBEDIENCE_SYSTEM Linear obedience constraints for the coordinator's recommendations.
%   [A, Pm, info] = obedience_system(c1, c2). Each row is one constraint "following the
%   recommendation r is no worse than deviating to d", flattened row-major over the 3x3
%   joint distribution. Rows: player 1 with (r, d), then player 2 with (r, d), d ~= r.
%   Pm holds the matching probability-mass coefficients; info(j,:) = [player r d], one-based.
A = zeros(12, 9);
Pm = zeros(12, 9);
info = zeros(12, 3);
j = 0;
for player = 1:2
    for r = 1:3
        for d = 1:3
            if d == r
                continue
            end
            j = j + 1;
            m = zeros(3);
            pm = zeros(3);
            if player == 1
                m(r, :) = c1(r, :) - c1(d, :);
                pm(r, :) = 1;
            else
                m(:, r) = c2(:, r) - c2(:, d);
                pm(:, r) = 1;
            end
            A(j, :) = reshape(m.', 1, []);
            Pm(j, :) = reshape(pm.', 1, []);
            info(j, :) = [player, r, d];
        end
    end
end
end
