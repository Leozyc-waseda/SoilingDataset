function tprint(arg)

if strcmp(arg,'start')
    tic;
elseif strcmp(arg,'stop')
    t = toc; fprintf('\t\t<time %f>\n',t);
else
    error('Invalid selection in timer');
end