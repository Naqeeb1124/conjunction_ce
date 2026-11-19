function run_tests(pattern)
%RUN_TESTS Run the test files in tests/ and print one PASS/FAIL line per check.
%   run_tests              runs every tests/test_*.m
%   run_tests('test_core.m')   runs the files matching the pattern
%   The last line is "SUMMARY: <n> passed, <m> failed".  In batch mode the exit code is 1
%   if any check failed.
root = setup_paths();
if nargin < 1
    pattern = 'test_*.m';
end
files = dir(fullfile(root, 'tests', pattern));
check_log('reset');
for i = 1:numel(files)
    [~, name] = fileparts(files(i).name);
    try
        feval(name);
    catch err
        check_true([name ' ran to completion'], false, strrep(err.message, newline, ' '));
    end
    close all force
end
[n_pass, n_fail] = check_log('count');
fprintf('SUMMARY: %d passed, %d failed\n', n_pass, n_fail);
if n_fail > 0 && batchStartupOptionUsed
    exit(1);
end
end
