function P = load_params()
root = setup_paths();
P = jsondecode(fileread(fullfile(root, 'params.json')));
end
