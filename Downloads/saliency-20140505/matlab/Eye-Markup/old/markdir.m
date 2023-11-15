function markdir(expdir,varargin)

if ((expdir(end) ~= '/') && (expdir(end) ~= '\'))
    expdir = [expdir,'/'];
end

dr = dir(expdir);
c = 1;
for (ii = 1:length(dr))
    %if a directory enter into it and get the subject data.
    if (dr(ii).isdir && (dr(ii).name ~= '.') && (dr(ii).name ~= '..') )
        dl = [];
        disp(['Got data for: ',dr(ii).name]);
        markeye([expdir,dr(ii).name,'/','*.eyec'],'autosave',1);
    end
end        
