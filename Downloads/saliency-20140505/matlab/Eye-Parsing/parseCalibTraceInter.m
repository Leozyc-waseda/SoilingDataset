function [x1, y1, x2, y2] = parseCalibTraceInter(data)
%function [x1 y1 x2 y2] = parseCalibTraceInter(data)
%
% data should be a 3xN matrix, with first row = raw eye tracker x,
% second = raw eye tracker y, and third = pupil diameter. It should
% have been cleaned up with cleanTrace()

% temporal window where saccade typically occurs:
T_SACWIN = [250 400];

% required minimum fixation length (for the target only):
T_FIXLEN = 70;   % use 200 to be strict, 100 to be lenient

% max acceptable variation over a fixation, in calibrated pixels:
PIX_FIXMAX = 15;   % use 7 to be strict, 10 to be more lenient

fixdata = findFixations(data);

% if could not find fixations, abort:
if (fixdata(1,1) == -1 | size(fixdata, 2) < 2)
  x1 = -1; y1 = -1; x2 = -1; y2 = -1; disp('data too trashy - not used');
else
  % let's find the longest fixation period before the window; start
  % time must be strictly before the start of the window, but it's
  % okay if stop time is inside the window. Do the same for the
  % longest after the window:
  sz = size(fixdata);
  fixpre = -1 * ones(1, sz(1)); fixpost = -1 * ones(1, sz(1));
  
  for ii = 1:sz(2)
    if (fixdata(1, ii) < T_SACWIN(1) && ... % starts before start of window
	fixdata(2, ii) < T_SACWIN(2) &&... % ends before end of window
	fixdata(2, ii) - fixdata(1, ii) > fixpre(2) - fixpre(1)) % longest
      fixpre = fixdata(:, ii)';
    end
    
    if (fixdata(1, ii) > T_SACWIN(1) & ... % starts after start of window
	fixdata(2, ii) > T_SACWIN(2) & ... % ends after end of window
	fixdata(2, ii) - fixdata(1, ii) > fixpost(2) - fixpost(1)) % longest
      fixpost = fixdata(:, ii)';
    end
  end
  
  % get the fixation coordinates for return:
  x1 = fixpre(5); y1 = fixpre(8); x2 = fixpost(5); y2 = fixpost(8);

  if (x1 == -1 | x2 == -1) %check again to ensure fixation is not bogus
    x1=-1; y1=-1; x2=-1; y2=-1; 
    disp(['fixations too unstable - not used']);
  elseif ( fixpost(2) - fixpost(1) < T_FIXLEN) %target fixation is still
                                               %too short
    x1=-1; y1=-1; x2=-1; y2=-1; 
    disp('target fixation too short - not used');
  elseif any(isnan([x1 x2 y1 y2]))
    x1=-1; y1=-1; x2=-1; y2=-1; 
    disp('not enough fixations found - not used');
    
  else
        
    % ok, let's plot the means over the pre and post fixations:
    rangepre = fixpre(1):fixpre(2);
    rangepost = fixpost(1):fixpost(2);
    
    % let's make sure the variances are low:
    vari = max([ std(data(1:2,rangepre ), 0,2), 
		 std(data(1:2,rangepost), 0,2)]);
    if (vari > PIX_FIXMAX)
      disp('Fixations too variable - not used');
      x1=-1; y1=-1; x2=-1; y2=-1;
    else
      % display fixations as magenta x's
      data2 = NaN(size(data));
      data2(1, rangepre ) = x1; data2(2, rangepre ) = y1;
      data2(1, rangepost) = x2; data2(2, rangepost) = y2;

      plotTrace(data2, 'm-x');
    end
  end
end

if any(0==[x1,x2,y1,y2])
    x1 = -1; y1 = -1; x2 = -1; y2 = -1;  
    disp('Fixations lose tracking (zero) - not used');
end

%disp([ 'Raw endpoint coordinates (' num2str(x2) ', ' num2str(y2) ')' ]);

if ~any(-1==[x1 x2 y1 y2]) & all(~isnan([x1 x2 y1 y2]))    % good fixation 
  in = input('Use found fixations [RET=yes]? ', 's');
  if (length(in) > 0 & (in(1) == 'n' | in(1) == 'N'))
    x1 = -1; y1 = -1; x2 = -1; y2 = -1;  
  end
else
  in = input('Press [RET] to continue', 's');
end
