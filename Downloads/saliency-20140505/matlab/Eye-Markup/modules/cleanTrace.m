function result = cleanTrace(data,params)
% function result = cleanTrace(data)
% Clean a trace, putting NaN into x,y where blinks or other weird
% events occur
% written: Sept 2006
% author : David J. Berg
%modified: John Shen (Oct 2009)
%iLab - University of Southern California
%**************************************************************************

% get parameters
hsiz = getvalue('lot_width',params);  % half horizontal clean size
pd_range = getvalue('pup_range',params);
vel_max = getvalue('max_vel',params);
dpd_max = getvalue('pup_maxstep',params);
vblevel = getvalue('verbose',params);
V = getvalue('verboselevels',params);
% if pupil is too big or too small, mark it (note: NaNs always
% return 0)

is_smallpup = data.pd < min(pd_range);
is_largepup = data.pd > max(pd_range);
is_badpup = is_smallpup | is_largepup;

% if tracking is too jerky, mark it
is_toofast = data.vel > vel_max | [(abs(diff(data.pd)) > dpd_max) 0];

% if tracking hits zero or does not change over some samples, mark it
is_zero = all(data.xy==0,1);

% This seems to be an ISCAN specific-issue:
% The position that occurs when a pupil is totally lost 
% (when the pupil is unrealistically small)
% will be the same, and any points at that area will be bad.

LOT_pos = unique(data.xy(:,is_smallpup)', 'rows');
if(numel(LOT_pos)==2) % if there is only one still-coordinate
  % mark all of those positions as bad
  if vblevel>=V.SUB
      fprintf('\tISCAN Loss of Tracking position found: (%4.1f, %4.1f)\n', ...
           LOT_pos(1),LOT_pos(2));
  end
  is_badpup = is_badpup | ismember(data.xy',LOT_pos,'rows')';
end

% gather all bad tracking marks
badseed = find(is_badpup | is_toofast | is_zero);

% construct events in window before and after bad-tracking sites
badstart = max(badseed - hsiz, 1);
badstop = min(badseed + hsiz, data.len);
is_bad = makeRanges(badstart,badstop);

%mark those bad windows as NaN
result = data; 
result.xy(:,is_bad)=NaN;
result.vel (is_bad)=NaN;
result.pd  (is_bad)=NaN;

if vblevel>=V.SUB
    [foo bar] = getBounds(is_bad);
    fprintf('\t%d samples in %d tracks cleaned from trace\n', sum(is_bad), length(foo));
end