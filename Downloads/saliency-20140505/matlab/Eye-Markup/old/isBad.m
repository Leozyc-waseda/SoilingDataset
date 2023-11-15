function [b,btime,fixtime,mbtime] = isBad(data,thresh,varargin)
varout = parseinputs(varargin);
sr = getvalue('Sf',varout);
ss = getvalue('screen-size',varout);
boarder = 10;

%1
%fixation length check--to much sleeping?
[fe,fb] = findevent(data,0);
fixtime = (fb-fe);
fixtime = fixtime./sr;

%2
%too much offscreen
sbad = find((data(1,:)    > (ss(1)-boarder)) | ...
            (data(1,:) < (0+boarder)) | ...
            (data(2,:) > (ss(2)-boarder)) | ...
            (data(2,:) < (0+boarder)));


%3
ff = find((data(3,:) ==2) | (data(3,:) == 3));
%see how much bad, and or blinking


btime = length(sbad)./length(data);
fixtime = find(fixtime > 5);
mbtime = length(ff)./length(data);

if ( (btime > thresh) || (mbtime > thresh) || ~isempty(fixtime) )
  
    b = 1;
else
    b = 0;
end