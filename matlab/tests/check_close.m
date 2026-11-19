function ok = check_close(name, got, want, rtol, atol)
%CHECK_CLOSE Check that |got - want| <= atol + rtol * |want| holds for every entry.
%   Vectors are compared regardless of orientation.  The printed line gives the largest
%   absolute error and the largest relative error over the nonzero reference entries.
if nargin < 5
    atol = 0;
end
got = double(got);
want = double(want);
if isvector(got) && isvector(want) && numel(got) == numel(want)
    got = got(:);
    want = want(:);
end
if ~isequal(size(got), size(want))
    ok = check_true(name, false, sprintf('size %s, expected %s', mat2str(size(got)), mat2str(size(want))));
    return
end
err = abs(got - want);
tol = atol + rtol * abs(want);
nonzero = want ~= 0;
if any(nonzero(:))
    rel = max(err(nonzero) ./ abs(want(nonzero)));
else
    rel = 0;
end
within = err <= tol;
ok = check_true(name, all(within(:)) && all(isfinite(got(:))), ...
    sprintf('max abs err %.3g, max rel err %.3g (rtol %g, atol %g)', max(err(:)), rel, rtol, atol));
end
