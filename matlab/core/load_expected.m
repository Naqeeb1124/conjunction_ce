function E = load_expected()
root = setup_paths();
E = jsondecode(fileread(fullfile(root, 'expected.json')));
end
