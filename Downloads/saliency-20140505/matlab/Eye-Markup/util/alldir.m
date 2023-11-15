function filelist = alldir(glob)
% function filelist = alldir(glob)
% Given a glob, give a cell array with all the files 
% captured by a 'dir' or 'ls'
% For PC: works only with one standard wildcard (*.m), returns relative
% path
% For Unix/MAC: works with any wildcards, returns full path
% 
% Possible improvements may include symbolic link following, 
% regexps if necessary

if ispc % can't use system calls
    filesinfo = dir(glob); % this will not work with multiple wildcards
    % may not work with version < 7.1 (just a condensed for loop)
    
    if isempty(filesinfo)
        error('markEye:noFile','%s not found.',glob);
    end
    % puts the path back on the files from dir 
    % TODO: remove implicit assumption that all files are in same dir.
    glob_path = fileparts(glob);
    filelist = cell(1,length(filesinfo));
    for i = 1:length(filesinfo)
        filelist{i} = fullfile(glob_path, filesinfo(i).name);
    end
else
    % this can deal with multiple asterisks
    % also possibly regexps/symbolic links (untested)
    [is_err,filez]=system(['find ' glob ' -type f']); 
    if is_err~=0
        error('markEye:noFile','%s not found', glob);
    end
    filez = textscan(filez,'%s');
    filez = filez{1};
    filelist = cell(1,length(filez));
    for i = 1:length(filelist)
        % gets full absolute path, which may help
        [is_err filelist{i}] = system(['readlink -f ' filez{i}]); 
        filelist{i} = filelist{i}(1:end-1); % "chomps" \n
    end
end