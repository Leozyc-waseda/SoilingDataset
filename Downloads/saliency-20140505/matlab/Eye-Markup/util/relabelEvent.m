function result = relabelEvent(data, event_num, stat_new)
%Utility function to relabel events in the data by referring to
%the event indices.
%Event_num can be a list of indices or a logical array over indices
%NB: does not update event statistics, 
%and assumes events structure is updated
%written: Nov 2009
%author : John Shen
%iLab - University of Southern California
%*****************************************************************

result = data;
if isempty(event_num), return; end; % no events to relabel
if all(event_num==0), return; end;

start = cell2mat({data.events(event_num).onset});
stop  = cell2mat({data.events(event_num).offset});
ind = makeRanges(start,stop);
result.status(ind) = stat_new;

