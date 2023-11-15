function markBad(glob,thresh,fnam,varargin)
%tells if a set of eyeS files has too much loss of tracking or blink.
%
%inputs:    glob: dir to eyeS file[s](make sure to add *.eyeS or *.eye
%           if glob is a directory)
%
%optionals inputs:
%           thresh [0.5]: ratio of bad eyedata to be consideres as bad file
%           fnam [badfiles] : output filename for bad files
%           varargin: see 'help parseinputs' for all command line options
%
%
%written: Sept 2006
%author : David Berg
%Ilab - University of Southern California
%**************************************************************************
if (nargin < 1)
    disp('Error: At least a path/filename is needed.');
end
if (nargin < 2)
    thresh = .5;
end
if isempty(thresh)
  thresh = .5;
end
if (nargin < 3)
  fnam = 'badfiles';
end


varout = parseinputs(varargin);
ss = getvalue('screen-size',varout);

[filez,glob] = strip_file_path(glob);

ferp = fopen(fnam,'w');
%loop over our files
for ii = 1:length(filez)
    fnam = filez{ii};
    fnam = [glob,fnam];       
    [data,pupil] = loadCalibTxt(fnam); %load the file
    [b,btime,ftime,mbtime] = isBad(data,thresh,varargin);
    if (b)
        disp(fnam);
	disp(['Boarder: ',num2str(btime)]);
	disp(['Fixation Time: ', num2str(length(ftime))]);
	disp(['Marked Bad: ', num2str(mbtime)]);
	disp('');
        fprintf(ferp,'%s\n',fnam);
    end
    
end
fclose(ferp);
