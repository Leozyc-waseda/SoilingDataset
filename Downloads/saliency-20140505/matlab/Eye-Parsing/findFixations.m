function result = findFixations(data)
% function result = findFixations(data)
% finds fixations and fits a line to each 
% result has: start, stop, offsetx, slopex, meanx, offsety, slopey,
% meany, one column per fixation

SIG = 20;  % sigma of low-pass filter used to smooth data and get velocity
TH = 0.5;  % max velocity threshold for fixations: 0.3 is strict, 0.5 is lenient

% let's construct a wide edge filter
hw = floor(SIG * sqrt(-2 * log(0.001))); % half filter width
fil = -hw:1:hw; fil = exp(-fil.*fil/(2*SIG*SIG));
fil = fil / sum(fil);  % normalized Gaussian filter
fil = convClean(fil, [-1 0 1]); % do a sobel on this

% apply edge filter to get velocities
dx = abs(convClean(data(1, :), fil));
dx([1:hw end-hw+1:end])=0;
dy = convClean(data(2, :), fil);
dy([1:hw end-hw+1:end])=0;
vel = abs(dx.*dx + dy.*dy);

%split saccades and fixations according to fixation
sac = find(vel > TH);
fix = find(vel <= TH);

fixations = data; saccades = data;
fixations(:, sac) = NaN; saccades(:, fix) = NaN;
plotTrace(saccades, ':', 1); plotTrace(fixations, '-');

% find start/stop of fixations:
df = diff(fix); % diff finds discrete derivative
fstart = [fix(1) fix(find(df~=1)+1)];
fstop =  [fix(find(df~=1)) fix(end)];

% if not at least 1 fixation, stop right here:
if (isempty(fstart))
  result = -1 * ones(8, 1);
  return;
end

% eliminate fixations that are too short:
MIN_FIXT = 50; %in samples
too_short = find(fstop - fstart < MIN_FIXT);
fstart(too_short) = [];
fstop(too_short)  = [];

% result has: start, stop, offsetx, slopex, meanx, offsety, slopey, meany

result = zeros(8, length(fstart));
result(1, :) = fstart; result(2, :) = fstop;
fixfits = NaN(size(data));

% fit a line to the fixations, in x and y:
for ii = 1:length(fstart)
  range = fstart(ii):fstop(ii);

  % fit line to x
  p = polyfit(range, data(1, range), 1);
  pval = polyval(p, range);
  result(3:5, ii) = [p(2), p(1), mean(pval)]; 
  fixfits(1, range) = pval;
  
  % fit line to y
  p = polyfit(range, data(2, range), 1);
  pval = polyval(p, range);
  result(6:8, ii) = [p(2), p(1), mean(pval)]; 
  fixfits(2, range) = pval;
end

% plot fits
plotTrace(fixfits, 'r--');
