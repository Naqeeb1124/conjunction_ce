function varargout = check_log(command, ok)
%CHECK_LOG Counter of passed and failed checks, shared by the check_* functions.
%   check_log('reset')            start counting from zero
%   check_log('add', ok)          record one check
%   [n_pass, n_fail] = check_log('count')
persistent n_pass n_fail
if isempty(n_pass)
    n_pass = 0;
    n_fail = 0;
end
switch command
    case 'reset'
        n_pass = 0;
        n_fail = 0;
    case 'add'
        if ok
            n_pass = n_pass + 1;
        else
            n_fail = n_fail + 1;
        end
    case 'count'
        varargout = {n_pass, n_fail};
    otherwise
        error('check_log:command', 'Unknown command "%s".', command);
end
end
