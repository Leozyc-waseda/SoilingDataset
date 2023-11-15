function result = filterTrace(data, params,filtsize)
% This returns a smoothened trace in both x, y, and the velocity
% The smoothing is done by a low-pass zero-phase Butterworth filter. 
sf = getvalue('sf',params);

if nargin<3
  error('filterTrace:badArgs','No filter size given');
elseif(filtsize == 0)
  return;
end

result = data;
% smooth the trace:

ff = filtsize/sf;%create the filter in Normalized freq units
[b a] = butter(4,ff,'low');
% filter and convert to deg per second
result.xy(1, :) = filtfilt(b,a,data.xy(1,:));
result.xy(2, :) = filtfilt(b,a,data.xy(2,:));
result.vel=getVel(result,params); % update velocity

