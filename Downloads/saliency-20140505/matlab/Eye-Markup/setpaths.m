% SETPATHS
% find the Eye-Markup folder with markEye.m
% and add the required paths
rootdir = fileparts(which('markEye')); 
addpath(rootdir);

extradirs = {'util', 'gui', 'plot', 'modules', 'user'};
for ii = 1:length(extradirs)
    addpath(fullfile(rootdir, extradirs{ii}));
end