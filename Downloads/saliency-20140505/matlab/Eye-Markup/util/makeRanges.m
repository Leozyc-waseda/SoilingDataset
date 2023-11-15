function b = makeRanges(eventbeg, eventend, opt)
% MAKERANGES logical which is true only between events
%
%   B = MAKERANGES(eventbeg, eventend, opt): returns a logical that is true
%   between eventbeginnings and endings, including both
%   the beginning and the end closed intervals, basically
%   opt can be 'closed', 'open', 'left-open', 'right-open'
%   default is 'closed'

% written: John Shen (Nov 2009)
% iLab - University of Southern California
%**************************************************************************

if nargin<2
    error('makeRanges:nomatch', 'Missing arguments for markers');
elseif nargin==2
    opt = 'closed';
end

switch opt
    case {'closed','left-open'} %include eventends
        eventend = eventend + 1;
    case {'left-open', 'open'} %exclude eventbegs
        eventbeg = eventbeg + 1;
end

if(length(eventbeg) ~= length(eventend))
    error('makeRanges:mismatch', 'Mismatched lengths for markers');
elseif(any(eventbeg>=eventend))
    error('makeRanges:badmatch', 'Events are not aligned');
elseif(isempty(eventbeg))
    b = [];
    return;
end

% trick for making overlapping ranges efficiently
% imagine b is depth of an expression - when depth is at 0 you are
% in the background
len = 1:eventend(end);
b = hist(eventbeg,len)-hist(eventend, len);
b = (cumsum(b)>0);
