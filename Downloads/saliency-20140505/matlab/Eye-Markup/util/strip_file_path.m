function [filez,glob] = strip_file_path(glob)
%strips the file names and path from a string identifying a group of files.
%
%INPUTS
%glob : path to file names with an extension (i.e. *.txt)
%
%OUTPUTS
%filez: a struct containing each file in alphabetical order
%glob : the path name stripped of the extension ending in a '/'or '\';
%
%Ex: [f,g] = strip_file_path('../test/*.txt');

%get just the filenames
if (iscell(glob))
    for (ii = 1:length(glob))
        gl = glob{ii};
        %a silly thing so the script works in windows and linux
        f = findstr(gl,'\');
        if (isempty(f))
            f = findstr(gl,'/');
        end
        filez{ii} = gl(f(end)+1:end);
        gl(f(end)+1:end) = [];
    end
    glob = gl;
else   

    filz = dir([glob]);
    for (ii = 1:length(filz))
        filez{ii} = filz(ii).name;
    end
    %a silly thing so the script works in windows and linux
    %NB: filesep can detect system dependent slashes
    % f = findstr(glob,filesep);
    f = findstr(glob,'\');
    if (isempty(f))
        f = findstr(glob,'/');
    end
    glob(f(end)+1:end) = [];
end
