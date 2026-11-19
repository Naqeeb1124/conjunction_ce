function ok = check_true(name, condition, detail)
%CHECK_TRUE Record one check and print it as a single line: PASS or FAIL, name, detail.
if nargin < 3
    detail = '';
end
ok = ~isempty(condition) && all(logical(condition(:)));
check_log('add', ok);
if ok
    tag = 'PASS';
else
    tag = 'FAIL';
end
fprintf('%s  %s  %s\n', tag, name, detail);
end
