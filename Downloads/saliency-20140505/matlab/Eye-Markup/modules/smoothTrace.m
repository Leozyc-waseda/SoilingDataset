function result = smoothTrace(data, params)
% This returns a smoothened trace in both x, y, and the velocity
% The smoothing is simple windowing over a binomial filter
% written: Oct 2009
% author : John Shen
% iLab - University of Southern California
%**************************************************************************

sf = getvalue('sf',params);
winsize = getvalue('smooth_window',params);

result = data; 

% creates a binomial filter 
%(e.g. for winsize, 7 would be [1 6 15 20 15 6 1]/64
ff = diag(rot90(pascal(winsize))); ff = ff' / sum(ff);

% do not smooth NaN regions
[start stop] = getBounds(all([~isnan(result.xy); ~isnan(result.pd)]));

for ii = 1:length(start)
  range = start(ii):stop(ii);
  result.xy(1, range) = convClean(data.xy(1, range), ff,'padded');
  result.xy(2, range) = convClean(data.xy(2, range), ff,'padded');
  result.pd(range) = convClean(result.pd(range), ff,'padded');
end

