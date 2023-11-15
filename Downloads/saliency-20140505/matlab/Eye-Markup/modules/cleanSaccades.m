function result = cleanSaccades(data,params)
%this function takes results(from reading an eye or eyeS file), and
%combines two saccades if they have a fixation of less then time and
%maintain the same trajectory.  It also removes saccades that are too
%short or have too small an amplitude, this should be determined from
%behavioral data.
%
%inputs: result: read from an eye or eyeS file
%
%        see help parsinputs for perameter options
%
%output: a newly marked result, which can be written to an eyeS file.
%
%written: Sept 2006
%author : David Berg
%modified: John Shen (Oct 2009)
%iLab - University of Southern California
%**************************************************************************

V = getvalue('verboselevels',params);
vblevel = getvalue('verbose',params);

sf = getvalue('sf',params);
sf = 1000./sf; %time between points

%for saccade combination
min_fixtime = getvalue('pro_timethresh',params);
min_fixtime = ceil(min_fixtime./sf); %time converted to samples 

angle_thresh = getvalue('pro_anglethresh',params);
resid_thresh = getvalue('pro_linearthresh',params);

%after we clean up, any saccade < sac_minamp is removed as it probably did
%not drive attention
%minampl = getvalue('sac_minamp',params) * ppd;
min_sacampl = getvalue('sac_minamp',params);

%for cleaning up short saccades
min_sactime = getvalue('sac_mintime',params);
min_sactime = ceil(min_sactime./sf);
scode = getvalue('code',params);

result = data;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is where we start the combine saccade code
if(~isfield(data,'events'))
  data.events = updateStats(data,params);
end

e = data.events;

% find all the fixation regions
is_fix = cell2mat({e.type})==0;
% find all regions shorter than a certain time
is_short = cell2mat({e.dur}) < min_fixtime;

% now find events that are sandwiched by saccades
% NB: intention of previous code (but not effect) was probably to see 
% if saccades were before and after an event in a small window.  
s_sac = [scode.SACCADE, scode.SACBLINK];
is_sac = ismember(cell2mat({e.type}),s_sac);
is_bordered = (convClean(+is_sac,[1 0 1])==2);

fix_to_combine = find(is_bordered & is_short & is_fix);

% find saccades that may not align well
is_combo = ones(1,length(fix_to_combine));
for ii = 1:length(fix_to_combine)
  reg = fix_to_combine(ii);
  sac_b = e(reg-1);
  sac_a = e(reg+1);

  % if saccades are relatively straight and aligned (this can be vectorized)
  is_combo = is_combo & ...
      RSSE(data.xy(:,sac_b.onset:sac_b.offset)) <= resid_thresh & ...
      RSSE(data.xy(:,sac_a.onset:sac_a.offset)) <= resid_thresh & ... 
      abs(sac_b.angle - sac_a.angle) <= angle_thresh;
end

% label combined saccades
fix_to_combine = fix_to_combine(is_combo);

result = relabelEvent(result, fix_to_combine, scode.SAC_CMBND);
e = updateStats(result,params);

% filter saccades that are too small in amplitude or duration
s_sac = [scode.SACCADE, scode.SACBLINK, scode.SAC_CMBND];
is_sac = ismember(cell2mat({e.type}),s_sac);
is_short_sac = is_sac & cell2mat({e.amp}) < min_sacampl; 
is_brief_sac = is_sac & cell2mat({e.dur}) < min_sactime;

sac_to_unmark = find(is_brief_sac | is_short_sac);

result = relabelEvent(result, sac_to_unmark, scode.NC);
e = updateStats(result,params);

% one last thing to do, if any saccade is before or after a junky
% spot then lets just mark it as junky as well, as it messes up our
% main sequence
% it's simpler to just iterate by region...

num_junky = 0;
s_junky = [scode.SACBLINK, scode.SAC_CMBND];
for ii = 2:length(e)-1
  if e(ii).type == scode.SACCADE
    if ismember(e(ii-1).type,s_junky);
      result = relabelEvent(result,ii, e(ii-1).type);
      num_junky=num_junky+1;
    elseif ismember(e(ii+1).type,s_junky);
      result = relabelEvent(result,ii, e(ii+1).type);
      num_junky=num_junky+1;
    end
  end
end

% report results
if vblevel >= V.SUB 
  fprintf(1, '\t%d combined fixations\n', length(fix_to_combine));
  fprintf(1,'\t%d saccades < %d degrees removed: sac-minamp criteria\n', ...
	  sum(is_short_sac), min_sacampl);
  fprintf(1,'\t%d saccades < %dms removed: sac-mintime criteria\n', ...
	  sum(is_brief_sac), min_sactime);
  fprintf(1, '\t%d saccades remarked for junkiness (proximity to bad saccades) \n', num_junky);
end

%%%%%%%%%%%%%%%%%%%%%%%
function rsum = RSSE(xy)
% returns residual sum of squared errors of a linear fit to
% data in (x,y)
if numel(xy)==2
    rsum = 0; % can't fit a line to a point
else
    % clear repeats 
    xy = unique(xy','rows');
    x = xy(:,1);
    y = xy(:,2);
    if numel(unique(x)) > 1 % can't fit w/ only one x coord
        eq = polyfit(x,y,1);
        rsum = norm(polyval(eq,x)-y);
    else
        rsum = std(y);
    end
end

