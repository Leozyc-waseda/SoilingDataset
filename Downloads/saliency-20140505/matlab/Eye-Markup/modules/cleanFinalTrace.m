function result = cleanFinalTrace(data,c_order,params)
%This function merges short marked areas into other areas.  It is
%written to replicate the behavior of cleanfinaltrace,  but is
%cleaner and demonstrates how the scheme may be modified
%
% The rules for cleaning are as follows:
% Only regions shorter than min_time are remarked.
% Regions are first marked if they are bracketed by a different
% region of the same type (fixation/saccade, roughly).
% Regions are then marked to their neighbor of the same type, if 
% they exist.  Rules for remarking are hard-coded in this routine.
% 
% inputs:    result : the data premarked by other files
%            c_order : which regions will be remarked first
%            see help parseinputs for more options
%
%
%written: Nov 2009
%author : John Shen
%iLab - University of Southern California
%*****************************************************************

min_time = getvalue('clean_window',params); %(ms)
sf = getvalue('Sf',params);
min_samples = ceil(min_time/1000*sf);
keepfix = getvalue('clean_keepfix',params);
scode = getvalue('code',params);
stat_allsac=[scode.SACCADE, scode.SACBLINK, scode.SAC_CMBND];

vblevel = getvalue('verbose',params);
V = getvalue('verboselevels',params);


% this should mimic the behavior of cleanfinaltrace - not a strict
% path by path translation
% nominals would make this code much clearer but not everyone has
% staistics toolbox
result=data;

% make sure that first and last events are at least of minimum length
% if not, merge the status of front events until they are of minimum length
evs_infirst = 1 + sum([data.events.offset]<min_samples);
if evs_infirst > 1
  front_ev = data.events(evs_infirst).type;
  result.status(1:min_samples)=front_ev;   
  if vblevel >= V.SUB
    fprintf('\tRelabeling first %d ms of events to %d\n',...
	    min_time, front_ev);
  end
end

evs_inlast = 1+sum([data.events.onset]>data.len-min_samples);
if evs_inlast > 1
  last_ev = data.events(end-evs_inlast).type;
  result.status(end-min_samples:end)=last_ev;   
  if vblevel >= V.SUB
    fprintf('\tRelabeling last %d ms of events to %d\n',...
	    min_time, last_ev);
  end
end

% rule 1: 
% if an event is short & sandwiched by two events of the same status, 
% convert to 2nd status
% replace short segments that are bracketed by fixations first 
% replace short segments that are bracketed by smooth pursuits afterwards 
stat_two_side = [scode.BLINK, scode.FIXATION; ...
		 scode.BLINK, scode.SMOOTH; ...
		 scode.SMOOTH, scode.FIXATION; ...
		 scode.FIXATION, scode.SMOOTH];

% NB: these jobs are done sequentially to show preference 
% to one status over another

for ii = 1:length(c_order)
  s_type = c_order(ii);
  
    % update event statistics
  result.events = updateStats(result,params);
  e = result.events;
    
  % do not relabel saccades
  rule_IDs = find(s_type==stat_two_side(:,1));
  if isempty(rule_IDs), continue; end;

  % catalog the info in arrays
  ev_dur = cell2mat({e.dur}); %in samples
  ev_type = cell2mat({e.type});
  ev_type_before = [-1 ev_type(1:end-1)]; % shift to the right
  ev_type_after  = [ev_type(2:end) -1  ]; % shift to the left

  % candidates for relabeling are short segments
  is_shorttype = (ev_dur < min_samples) & (ev_type==s_type);
  
  for jj = 1:length(rule_IDs)
    stat_replace = stat_two_side(rule_IDs(jj),2);
    is_bound = (ev_type_before == stat_replace ...
		    & ev_type_after == stat_replace);
  
    is_short_bound = is_shorttype & is_bound;  
    result = relabelEvent(result, is_short_bound, stat_replace);
    if any(is_short_bound) && vblevel >= V.SUB 
      fprintf('\t%d short events remarked: %d->%d\n', ...
	      sum(is_short_bound),s_type,stat_replace);
    end
  end
  % update event statistics
  result.events = updateStats(result,params);
  e = result.events;

end

% rule 2:
% if the first event is short & adjacent to the second status, 
% convert to 2nd status
% rule 3: 
% if the event is short and is nothing special (fixation or
% smooth), relabel it back
stat_one_side = [scode.SACCADE, scode.SACBLINK; ...
		   scode.SAC_CMBND, scode.SACBLINK; ...
		   scode.SMOOTH, scode.FIXATION; ...
		   scode.FIXATION, scode.SMOOTH];
stat_to_drift = [scode.FIXATION; scode.SMOOTH];

for ii = 1:length(c_order)
  s_type = c_order(ii);
  
  % update event statistics
  result.events = updateStats(result,params);
  e = result.events;

  % do not relabel saccades
  rule_IDs = find(s_type==stat_one_side(:,1));
  if isempty(rule_IDs), continue; end;

  % catalog necessary info
  ev_dur = cell2mat({e.dur}); %in samples
  ev_type = cell2mat({e.type});
  ev_type_before = [-1 ev_type(1:end-1)]; % shift to the right
  ev_type_after  = [ev_type(2:end) -1  ]; % shift to the left

  % candidates for relabeling
  is_shorttype = (ev_dur < min_samples) & (ev_type==s_type);

  for jj = 1:length(rule_IDs)
    stat_replace = stat_one_side(rule_IDs(jj),2);
    is_adj = (ev_type_before == stat_replace ...
		    | ev_type_after == stat_replace);
  
    is_short_adj = is_shorttype & is_adj;   
    is_shorttype = is_shorttype & ~is_adj; % keep track of ones not marked
  
    result = relabelEvent(result, is_short_adj, stat_replace);
    if any(is_short_adj) && vblevel >= V.SUB
      fprintf('\t%d short events remarked 2nd pass: %d->%d\n', ...
	      sum(is_short_adj),s_type,stat_replace);
    end
  end

  % do not remark any segments that are bordered by saccades
  is_desired = ismember(ev_type_before, stat_allsac) & ...
      ismember(ev_type_after,stat_allsac);
  is_drift = is_shorttype & ~is_desired;

  % mark the rest of the unmarked short events as drifts
  if(any(s_type == stat_to_drift))
    result = relabelEvent(result, is_drift, scode.NC);
    if any(is_drift) && vblevel >= V.SUB 
      fprintf('\t%d short events of %d remarked 2nd pass as drift\n', ...
	      sum(is_drift), s_type);
    end
  end
  
end