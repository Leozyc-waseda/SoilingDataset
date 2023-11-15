function [freqs,afreqs,wn] = spectral_analysis(glob,totalsize,chunksize,varargin)
%path and a chunksize< chunksize should be in ms
%so we want to find out some frequency information about the saccades, but
%a regular fft method wont be good as the eye movement signal is so non
%stationary.  %so lets do something like a baysean method.  and just ask for 
%each saccade what the likelihood of seeing a saccade in, x ms bins would be.
%here there is no overlap in the bins of saccade times.  
%a wavelet technique also comes to mind, and add up across translations.  

%returns the main sequence of some data
varout = parseinputs(varargin);
sf = getvalue('Sf',varout);
ppd = getvalue('ppd',varout);%for amp and velocity conversions
sacfilt = getvalue('sac-filter',varout);
scsz = getvalue('screen-size',varout);
wn = ceil(totalsize./chunksize);
wn = 0:totalsize/wn:totalsize;

afreqs = [];
[filez,glob] = strip_file_path(glob);
amp = [];
pvel = [];
cc = 0;
freqs  = zeros(size(wn));
for (ii = 1:length(filez))
  data = [];
  fnam = [glob,filez{ii}];
  data = loadCalibTxt(fnam); %load the file
  [sb,se] = findEvent(data,1,1);%find saccades not including
  time = (sb-1)*(1000/sf);%no saccade times in ms
  for (s = 1:length(time))
    
    ftime = (time-time(s));
    %we use a greater than just so we remoce the baseline, or zero bin
    ftime = ftime(find((ftime > 0) & (ftime <= totalsize)));
    freqs = freqs + histc(ftime,wn);
  
  end
  ii
  %lets also get an estimate of saccades per second while we are here.  
  spers(ii) = length(time) / ((length(data)-1)*(1/sf));

end
freqs = freqs./sum(freqs);
afreqs = spers;

%  if (isempty(freqs))
%    freqs = pwelch(time,wn)';
%  else
%    freqs = freqs + pwelch(time,wn)';
%  end
%  cc = cc+1;
%  ii
%end
%freqs = freqs./cc;
