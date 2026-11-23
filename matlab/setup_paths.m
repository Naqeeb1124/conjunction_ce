function root = setup_paths()
root = fileparts(mfilename('fullpath'));
addpath(root);
for sub = ["core", "twobody", "simulink", "viz", "tests"]
    folder = fullfile(root, sub);
    if isfolder(folder)
        addpath(folder);
    end
end
end
