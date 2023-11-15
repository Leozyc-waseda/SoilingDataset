function result = findSaccades(data,params)
%function [result,r] = findSaccades(data,params)
%marks the saccades in data, this uses simple filtering and instantaneous
%velocity to determine saccades
%inputs:    data: is the xy data in a var X samples matrix
%
%           see help parseinputs for more options
%           
%written: 
%author : Laurent Itti
%modified: David Berg (Sept 2006)
%modified: John Shen (Oct 2009)
%Ilab - University of Southern California
%********************************************************              

V = getvalue('verboselevels',params);
vblevel = getvalue('verbose',params);

sf = getvalue('sf',params);
sacfilt = getvalue('sac_filter',params);
minvel = getvalue('sac_minvel',params);% minimum amplitude of the velocity in degrees per sec
maxSacLength = getvalue('maxsaclength',params); %in ms
scode = getvalue('code',params);

result = data;

% NB: output does NOT get filtered
data = filterTrace(data,params,sacfilt);

% find saccades start/end:
insac = (data.vel > minvel);
[sacbeg sacend] = getBounds(insac);

% pad each to prevent saccades from colliding
sacbeg = [-Inf sacbeg data.len+1];
sacend = [1 sacend Inf];

% go over each saccade:
for ii = 2:length(sacbeg)-1
  % starting from the beginning of the saccade, move towards the
  % start as long as the velocity is decreasing and we don't bump
  % into the previous saccade; but enforce at least one sample that
  % will not be marked as saccade between the two::

  prev_fix = sacend(ii-1):sacbeg(ii);

  accl = [-Inf diff(data.vel(prev_fix))];
  % find where acceleration was last decreasing rather than increasing 
  % or at worst just pick the front of the window.
  p_infl = find(accl<0,1,'last');  
  sacbeg(ii) = prev_fix(p_infl); 

  next_fix = sacend(ii):sacbeg(ii+1)-1;
  accl = [diff(data.vel(next_fix)) Inf];
  
  % find where acceleration is last increasing rather than decreasing 
  % or at worst just pick the end of the window.
  p_infl = find(accl>0,1,'first');
  sacend(ii) = next_fix(p_infl); 
end

%unpad arrays
sacbeg = sacbeg(2:end-1);
sacend = sacend(2:end-1);

% go over all our saccades and mark them off:
for ii = 1:length(sacbeg)
  sacrange = sacbeg(ii):sacend(ii);
  % if any of the saccades are already in blink mark as sac+blink
  if (any(data.status(sacrange) == scode.BLINK))
    tmp = length(sacrange) * (1000/sf);
    if (tmp < maxSacLength) %if saccade is not too long?
      result.status(sacrange) = scode.SACBLINK;
    end
  else
    % mark the saccades as simple, but no maxSaclength?
    result.status(sacrange) = scode.SACCADE;
  end
end

if vblevel>=V.SUB
    is_sac = result.status == scode.SACCADE;
    [sb, se] = getBounds(is_sac);
    fprintf('\t%d saccades found with %d samples \n', length(sb), sum(is_sac));
    
    is_sacblink = result.status == scode.SACBLINK;
    [sbb,sbe] = getBounds(is_sacblink);
    fprintf('\t%d saccades in blink found with %d samples\n', length(sbb), sum(is_sacblink));
end

% run a simple pca and save the ratio of the principle axis for each window
% size 
% Also should make sure there is some variance, otherwise we probably have the
% same points,do an average of a couple of different sliding pca windows
% this effectively shortens the saccades to very straight lines

sacthresh = getvalue('sac_pcathresh',params);
winp = getvalue('sac_pcawindow',params); % window sizes in ms
winp = (winp ./ (1000/sf)); % window size in samples.  
[result,r] = pca_window(result,winp);

% if there is too much variance in a saccade region
is_ff = (r > sacthresh) & ismember(result.status,[scode.SACCADE, scode.SACBLINK]);
result.status(is_ff) = scode.FIXATION;

if vblevel>=V.SUB
    is_newsac = is_sac & ~is_ff;
    is_newsacblink = is_sacblink & ~is_ff;
    
    [sb, se] = getBounds(is_newsac);
    fprintf('After pca thresholding:\n');
    fprintf('\t%d saccades found with %d samples \n', length(sb), sum(is_newsac));
    
    [sbb,sbe] = getBounds(is_newsacblink);
    fprintf('\t%d saccades in blink found with %d samples\n', length(sbb), sum(is_newsacblink));
    
end
