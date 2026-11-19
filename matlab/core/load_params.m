function P = load_params()
%LOAD_PARAMS Decode params.json from the matlab/ folder.
%   P = load_params() returns the struct with fields constants, scenario, gallery, and labels.
root = setup_paths();
P = jsondecode(fileread(fullfile(root, 'params.json')));
end
