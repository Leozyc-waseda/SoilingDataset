function result = cleanTrace(data)
%function result = cleanTrace(data)
% Clean a trace, putting NaN into x,y where blinks or other weird
% events occur

hsiz = 5;  % half horizontal clean size

result = data; sz = size(data);

%if pupil is too big or too small, mark it
badpup = find(data(3, :) < 50 | data(3, :) > 1000);
zeroed = find(data(1,:) == 0 | data(2,:) == 0);

badpup = [badpup zeroed];
%construct events in window before and after abnormal pupil sizes
badpupstart = badpup - hsiz; badpupstart(find(badpupstart < 1)) = 1;
badpupstop = badpup + hsiz; badpupstop(find(badpupstop > sz(2))) = sz(2);

%mark those windows as NaN
for ii = 1:length(badpupstart)
  result(:, badpupstart(ii):badpupstop(ii)) = NaN;
end

