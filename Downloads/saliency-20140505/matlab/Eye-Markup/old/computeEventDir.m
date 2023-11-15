function computeEventDir(glob,order,varargin);
% function computeEventDir(glob,order,pv,varargin);
% glob: path of files
% order: event number, 1 for saccade, etc.

%compute some statistics about events
if (nargin < 1)
    disp('Error: At least a path/filename is needed.');
end

varargin;
varout = parseinputs(varargin);
ext = getvalue('compute-ext', varout);
pv = getvalue('save-peakvel',varout);

[filz,glob] = strip_file_path(glob);

%loop over our files
for ii = 1:length(filez)
    fnam = filez{ii};
    fnam = [glob,fnam];
    disp(['Loading [' fnam ']....']);
    [data,pupil] = loadCalibTxt(fnam); %load the file
    result = computeEvent(data,order,varout);
    %lets add the pupil back in
    result = [result(1:2,:);pupil;result(3:end,:)];
    dot = find(fnam == '.');    
    fnam = fnam(1:dot(end));
    fnam = [fnam,ext];
    fil = fopen(fnam, 'w');
    if (fil == -1), disp(['Cannot write ' fnam{ii}]); return; end
    if (pv == 1)
      fprintf(fil, '%.1f %.1f %.1f %d %.1f %.1f %.1f %d %.1f\n', result);
    else
      fprintf(fil, '%.1f %.1f %.1f %d %.1f %.1f %.1f %d\n', result(1:8,:));
    end
    
    fclose(fil);
    disp(['Wrote ' fnam]);
end
