function cuteyeDirectory(expdir,varargin)

if ((expdir(end) ~= '/') && (expdir(end) ~= '\'))
    expdir = [expdir,'/'];
end
cond = find((expdir == '/') || (expdir == '\'));
cond = expdir(cond(end-1)+1:cond(end)-1);

dr = dir(expdir);
c = 1;
for (ii = 1:length(dr))
    %if a directory enter into it and get the subject data.
    if (dr(ii).isdir && (dr(ii).name ~= '.') && (dr(ii).name ~= '..') )
        dl = [];
        disp(['Got data for: ',dr(ii).name]);
        cuteyefiles([expdir,dr(ii).name,'/','*.eye']);
    end
end        
