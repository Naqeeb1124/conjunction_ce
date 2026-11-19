function root = setup_paths()
%SETUP_PATHS Put the folders of the MATLAB layer on the path and return the matlab/ folder.
root = fileparts(mfilename('fullpath'));
addpath(root);
for sub = ["core", "twobody", "simulink", "viz", "tests"]
    folder = fullfile(root, sub);
    if isfolder(folder)
        addpath(folder);
    end
end
end
