function cuteyefiles(glob,samp)
if (nargin < 2)
    samp = 240;
end

d = dir(glob);

f = findstr(glob,'\');
if (isempty(f))
    f = findstr(glob,'/');
end
glob(f(end)+1:end) = [];

for (ii = 1:length(d))
    fname = d(ii).name;
    disp(['cutting: ',[glob,fname]]);
    data = loadCalibTxt([glob,fname]);
    data(:,1:samp) = [];
    dlmwrite([glob,fname,'c'],data',' ');    
end

