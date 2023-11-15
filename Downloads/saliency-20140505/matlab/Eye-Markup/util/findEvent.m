function [eventbeg,eventend] = findEvent(data,s_sel)
%[eventbeg,eventend] = findEvent(result,s_sel)
%find areas in question
%returns vectors of beginning and endpoints
%needs argument documentation.

params = parseinputs;
scode = getvalue('code',params);
s_sac = [scode.SACCADE, scode.SACBLINK, scode.SAC_CMBND];
s_validsac = [scode.SACCADE, scode.SAC_CMBND];

% zero-pad array 
f = [0 ismember(status, s_sel) 0];
df = diff(f);
eventbeg = find(df==1);
eventend = find(df==-1)-1;
