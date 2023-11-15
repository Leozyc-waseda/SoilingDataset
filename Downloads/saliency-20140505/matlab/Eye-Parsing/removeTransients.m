function result = removeTransients(data)
%function result = removeTransients(data)
% Clean a trace, putting NaN into x,y where blinks or other
% transients occur, then smooth the data

hsiz = getvalue(params,'cleanwidth');  % half horizontal clean size
txy = 50;   % threshold on the difference in x/y
tpd = 100;  % threshold on the difference in pupil diameter

result = data; sz = size(data);

df = abs(diff(data,1,2));
toofast = find(df(1,:) > txy | df(2,:) > txy | ...
	       df(3,:) > tpd);

badstart = toofast - hsiz; badstart(badstart < 1) = 1;
badstop = toofast + hsiz; badstop(badstop > sz(2)) = sz(2);
for ii = 1:length(badstart)
  result(:, badstart(ii):badstop(ii)) = NaN;
end

% smoothing window of width 7 
% can also be diag(rot90(pascal(7)))'
ff = [1 6 15 20 15 6 1]; ff = ff / sum(ff);
result(1, :) = convClean(result(1, :), ff,'padded');
result(2, :) = convClean(result(2, :), ff,'padded');
result(3, :) = convClean(result(3, :), ff,'padded');
