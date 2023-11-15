function [fixtimes,smoothtimes,intertimes] = intersaccadic(glob,varargin)
  

%returns the main sequence of some data
varout = parseinputs(varargin);
sf = getvalue('Sf',varout);
ppd = getvalue('ppd',varout);%for amp and velocity conversions
sacfilt = getvalue('sac-filter',varout);
scsz = getvalue('screen-size',varout);

[filez,glob] = strip_file_path(glob);
cc = 0;
fixtimes = [];
smoothtimes = [];
intertimes = [];

for (ii = 1:length(filez))
  data = [];
  fnam = [glob,filez{ii}];
  data = loadCalibTxt(fnam); %load the file
  
  [sb,se] = findEvent(data,0,1);%find fixation
  fixtimes = [fixtimes,(se-sb)*(1000/sf)];%no saccade times in ms
              
  [sb,se] = findEvent(data,4,1);%find smooth
  smoothtimes = [smoothtimes,(se-sb)*(1000/sf)];%no saccade times in ms
  
  [sb,se] = findEvent(data,1,1);%find intersaccadic
  sb(1) = [];se(end) = [];
  intertimes = [intertimes,(sb-se)*(1000/sf)];%no saccade times in ms
  
ii

end