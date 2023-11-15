% quick and dirty function to get rid of short saccades
function result = excludeBriefSaccades(data,args)
scode = getvalue('code', args);
% extreme is (5.62 318.7 50)
short_amp = 6; %degrees
sac_minlen = 80; % ms
too_fast = 100; % degrees/s

ev = data.events;
is_sac = [ev.type]==scode.SACCADE;
is_short = [ev.amp]<short_amp;
is_brief = [ev.interval]<sac_minlen;
is_speedy = [ev.pvel]>too_fast;

is_toobrief = is_sac & is_short & is_brief & is_speedy;
result = relabelEvent(data, is_toobrief, scode.FIXATION);
end