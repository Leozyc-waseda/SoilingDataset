function markBadDirectory(expdir,rr,varargin)

if (nargin < 2), rr = 0; end
varout = parseinputs(varargin);

if ((expdir(end) ~= '/') && (expdir(end) ~= '\'))
    expdir = [expdir,'/'];
end
cond = find((expdir == '/') || (expdir == '\'));
cond = expdir(cond(end-1)+1:cond(end)-1);

dr = dir(expdir);
c = 1;
for (ii = 1:length(dr))
    %if a directory enter into it and get the subject data.
    if (dr(ii).isdir && (~strcmp(dr(ii).name, '.')) && ...
	(~strcmp(dr(ii).name, '..')) )
        dl = [];
        disp(['Got data for: ',dr(ii).name]);
        markBad([expdir,dr(ii).name,'/','*.ceyeS'], rr, varout);
    end
end        
