function E = load_expected()
%LOAD_EXPECTED Decode expected.json from the matlab/ folder (reference numbers, not inputs).
%   E = load_expected() returns the struct with fields baseline, gallery, and source.
root = setup_paths();
E = jsondecode(fileread(fullfile(root, 'expected.json')));
end
