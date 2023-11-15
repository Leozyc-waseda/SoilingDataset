function result = eliminateBlinks(data,params)
%function result=eliminateBlinks(data,sampling freq)
% data has (x,y,pd) info and contains NaN when blinks
% occurred. Here we change the status so that it is 2 (default) 
% during blinks and 0 otherwise (fixation default). 
% We also interpolate the data across blinks.
%
% written: Sept 2006
% author : David J. Berg
% written: Nov 2009
% author : John Shen
% iLab - University of Southern California
%**************************************************************************

sf = getvalue('sf',params);
wind = getvalue('blink_window',params);
wind = ceil( wind/(1000/sf) ); %convert time to samples
scode = getvalue('code',params);

V = getvalue('verboselevels',params);
vblevel = getvalue('verbose',params);

% initialize results status
result = data;
result.status(:) = scode.FIXATION;

% compute locations of start/end of blink (NaN) regions:
[bstart bend] = getBounds(isnan(data.vel));

% exit if no blink at all:
if isempty(bstart) || isempty(bend)
  return;
end

% pad arrays for limiting purposes
bstart = [bstart data.len+1];
bend   = [bend Inf];

% find first point of lowest velocity past the blink end:
for ii = 1:length(bstart)-1
  
  % select the point of min velocity over our search window:
  % search window will end either at end of window, before start of
  % next blink, or at end of data

  b_after = bend(ii)+(1:wind); 
  b_after = min(b_after, bstart(ii+1)-1); % stop window before next blink
  
  tmp = data.vel(b_after);
  % else value is before the next blink
  pos = [b_after(tmp <= min(tmp)*1.01) b_after(end)];
  bend(ii) = pos(1);   % keep only first index

  % mark the blink:
  result.status(bstart(ii):bend(ii)) = scode.BLINK;    
end
bstart(end) = []; bend(end) = [];

% cubic interpolation over blinks (over NaN entries, more precisely)
t_all=1:data.len;
t_good=find(~any(isnan(result.xy),1));
xy_good=result.xy(:,t_good);

result.xy(1,:) = interpCubic(t_good,xy_good(1,:),t_all);
result.xy(2,:) = interpCubic(t_good,xy_good(2,:),t_all);

if vblevel>=V.SUB
    is_blink = makeRanges(bstart,bend);
    fprintf('\t%d samples in %d regions marked as blink, status %d\n', ...
            sum(is_blink),length(bstart),scode.BLINK);
end

function y_all = interpCubic(x_good,y,x_all)
% cubic interpolation with constant extrapolation
% x_good and x_all should be sorted - otherwise this will not work
% if you want to change this to match previous behavior, 
% just substitute 'linear' for 'cubic'

y_all = interp1(x_good,y,x_all,'cubic');

if x_good(1) ~= x_all(1)
  x_first = x_good(1);
  y_all(x_all<x_first) = y(1);
end
if x_good(end) ~= x_all(end)
  x_last = x_good(end);
  y_all(x_all>x_last) = y(end);
end

