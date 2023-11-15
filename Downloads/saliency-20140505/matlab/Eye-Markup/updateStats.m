function events = updateStats(data,params)
% Returns a list of regions of the saccade trace as a struct array.
% Calculates modelfree parameters for each event such as amplitude,
% angle, velocity, avg pupil diameter, duration, etc. as well as onset/offset
% and origin/target.
% Also computes time (in samples)to the next/prev event of a given
% type,
% encoded as an array:
% time_to_next[status_code+1] for status_code=0:6.
% and similarly for previous.
% Time to next and previous saccade is calculated here as well.
% This function replaces the need for computeEvent or modelfree in
% previous versions.

sf = getvalue('sf',params);
ppd = getvalue('ppd',params);%for amp and velocity conversions
%sacfilt = getvalue('sac_filter',params);
%scsz = getvalue('screen_size',params);
strict = getvalue('modelfree_strict',params);
%stats = getvalue('stats',params);
scode = getvalue('code',params);
alignstimulitime = getvalue('alignstimulitime', params);
if alignstimulitime 
    trash = double(getvalue('trash',params));
else
    trash = 0;
end

tw = getvalue('event_window', params);
point_halfwin = ceil((tw/2/1000) * sf);

% if there is are no status code, skip this function:
if all(isnan(data.status))
    events = struct([]);
    return;
end

% loop through entire time sequence - doesn't try to look at old events
% events will be a structure array of different events
% time units are in samples
samp_ptr = 1;
ii=1;

% preallocation
Nevents = sum(diff(data.status)~=0)+1;
events(Nevents).type = [];

%smf_f = getvalue('smf_prefilter',params); %low pass cutoff for smooth
%sac_f = getvalue('sac_filter',params); %low pass cutoff for fixation
%[foo,r_smf] = pca_clean(data,smf_f,winp,params);
%[foo,r_sac] = pca_clean(data,sac_f,winp,params);

while samp_ptr <= data.len
    % "initialize" event
    events(ii).type = data.status(samp_ptr);
    
    % find event endpoint after our pointer
    is_event_after = (data.status == events(ii).type) & (1:data.len)>=samp_ptr;
    
    % last 1 b/c the last frame is always an event offset
    toggle = [(diff(~is_event_after) == 1) 1];
    samp_off = find(toggle,1,'first');
    
    % mark temporal properties (in samples)
    events(ii).onset = samp_ptr;
    events(ii).offset = samp_off;
    
    % both bounds are included in the sample
    events(ii).dur = events(ii).offset - events(ii).onset + 1;
    
    events(ii).interval = events(ii).dur/sf*1000; % in msecs
    events(ii).timeon = (events(ii).onset-trash)/sf*1000;
    events(ii).timeoff = (events(ii).offset-trash)/sf*1000;
    
    events(ii).normedon  = events(ii).onset /(data.len-trash);
    events(ii).normedoff = events(ii).offset/(data.len-trash);
    
    % mark points of event start/stop
    on_win = samp_ptr+(-point_halfwin:point_halfwin);
    on_win(on_win<1 | on_win>data.len) = [];
    off_win = samp_off+(-point_halfwin:point_halfwin);
    off_win(off_win<1 | off_win>data.len) = [];
    
    % select the median of a small window as the points of origin
    events(ii).origin = median(data.xy(:,on_win),2);
    events(ii).target = median(data.xy(:,off_win),2);
    
    % how many events of this kind have transpired
    if ii == 1
        events(ii).typenum = 1;
    else
        events(ii).typenum = 1+sum([events(1:ii-1).type]==events(ii).type);
    end
    
    % result of pca analysis - may be useful for classification
    % events(ii).eigvar = mean(r_sac(range));
    % events(ii).eigvar = mean(r_smf(range));
    
    % if event is a blink or drift, then skip calculations...
    if any(events(ii).type == [scode.BLINK scode.NC])
        % NB: any new calculations have to be added to this list
        calcs = {'amp','angle','linangle','nonlin', 'pvel','avel','apup','spup'};
        for kk = 1:length(calcs)
            events(ii).(calcs{kk}) = NaN;
        end
    else
        % do derived calculations: amplitude in degrees,
        % peak/avg velocity in deg/sec, peak/avg pupil diameter
        range = samp_ptr:samp_off;
        
        % amplitude (in retinal degrees) and angle calculations (in degrees)
        r_disp = (events(ii).target-events(ii).origin)./ppd';
        events(ii).dx = r_disp(1)*ppd(1);
        events(ii).dy = r_disp(2)*ppd(2);
        events(ii).amp = norm(r_disp,2);
        
        % naive angle calculation (all angle ranges -180:180)
        events(ii).angle = rad2deg(atan2(r_disp(2),r_disp(1)));
        
        % angle calculation based on linear interpolation
        INTER_TOL = 1e-6;
        if numel(range) > 1 && all([events(ii).amp events(ii).dx ...
                events(ii).dy] > INTER_TOL)
            linefit = polyfit(data.xy(1,range),data.xy(2,range),1);
            
            % take tangent of slope to get angle, correcting for sign of vector
            events(ii).linangle = rad2deg(atan2(linefit(1)*sign(r_disp(1)),sign(r_disp(1))));
            
            % non-linearness as the root residual sum of squares
            events(ii).nonlin = norm(polyval(linefit,data.xy(1,range))-data.xy(2,range));
        else % there is no angle because there is almost no movement
            events(ii).linangle = NaN;
            events(ii).nonlin = 0;
        end
        % velocity stats (in ret degrees/sec)
        events(ii).pvel = max(data.vel(range));
        events(ii).avel = events(ii).amp/events(ii).dur*sf;
        
        % pupil stats
        events(ii).apup = mean(data.pd(range));
        events(ii).spup = std(data.pd(range));
    end
    
    % look at next unlabeled point in time
    samp_ptr = samp_off + 1;
    
    % go to next event
    ii=ii+1;
end
% OK, we've gone through one pass of the events.  now for inter-event
% statistics

% for indexing
scode_seq = [events.type];
index = 1:numel(events); % index of events

% for coding time_to_next and time_to_prev as arrays
code_names = fieldnames(scode);
Ncodes = numel(code_names);

% this dance makes sure that the following code works despite 
% any changes to what values are given as codes
% NB: we can also make time_to_next and time_to_prev structures in
% themselves, but I haven't tested the overhead for this in matlab

Ci = zeros(1,Ncodes);
s_ridx = scode;
for ii = 1:Ncodes
    Ci(ii) = scode.(code_names{ii}); % maps index 1->N to the code_value in
                                     % the order listed
    s_ridx.(code_names{ii}) = ii; % maps the code_name back to
                                      % the index 1->N used in time_to_prev
end

% this is what we consider saccades for time_to_prev/time_to_next saccade
allsac = [s_ridx.SACCADE, s_ridx.SACBLINK, s_ridx.SAC_CMBND];

time_to_prev = zeros(1,Ncodes);
time_to_next = zeros(1,Ncodes);
% calculate intervals between events (in samples)
for ii = 1:numel(events)
    for j = 1:Ncodes % across all statistics - should be flexible
        j_code = Ci(j); % 1:N --> code 
        
        ev_before = find(scode_seq == j_code & index < ii, 1, 'last');
        ev_after =  find(scode_seq == j_code & index > ii, 1, 'first');
        if(isempty(ev_before))
            time_to_prev(j) = NaN;
            % time_to_prev(j) = events(ii).onset; %also could be NaN
        else
            time_to_prev(j) = events(ii).timeon-events(ev_before).timeoff;
        end
        
        if(isempty(ev_after))
            time_to_next(j) = NaN;
            % time_to_next(jj) = data.len-events(ii).offset; %also could be NaN
        else
            time_to_next(j) = events(ev_after).timeon-events(ii).timeoff;
        end
    end
    events(ii).time_to_prev = time_to_prev;
    events(ii).time_to_next = time_to_next;
        
    % the +1s are to correct from 0->1 indexing
    events(ii).time_to_prev_saccade = min(time_to_prev(allsac));
    events(ii).time_to_next_saccade = min(time_to_next(allsac));
    if(strict) %only use proper saccades
        if(events(ii).time_to_prev_saccade ~= time_to_prev(s_ridx.SACCADE))
            events(ii).time_to_prev_saccade = NaN;
        end
        if(events(ii).time_to_next_saccade ~= time_to_prev(s_ridx.SACCADE))
            events(ii).time_to_next_saccade = NaN;
        end
    end
    
    if(ii>1 && scode_seq(ii-1) == scode.FIXATION)
        events(ii).bfix_dur = events(ii-1).interval;
    else
        events(ii).bfix_dur = NaN;
    end
    
    if(ii < Nevents && scode_seq(ii+1) == scode.FIXATION)
        events(ii).afix_dur = events(ii+1).interval;
    else
        events(ii).afix_dur = NaN;
    end
    
end

