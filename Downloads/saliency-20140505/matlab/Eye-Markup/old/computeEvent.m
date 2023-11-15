function result = computeEvent(result,ev,var)
%computes some statistics about the event in question
%stripped from find saccades will be updateded to handle a more general
%structure for any event and different types of info.  
%nowt hisd doesn't give the duration of the fixation after the saccade as
%there may not be one with the new markup, to be changed soon, now just the
%duration of the saccade
%can be called in batch from computeEventDir.m
%
%result: eye trace data already marked
%ev: event in question, for now only saccades
%pv: 1 for output peak velocity, others don't
%var: the normal list of string value pairs for settings

varout = parseinputs(var);
sf = getvalue('sf',varout);
ppd = getvalue('ppd',varout);%for amp and velocity conversions
pv = getvalue('save-peakvel',varout);
sacfilt = getvalue('sac-filter',varout)
scsz = getvalue('screen-size',varout)
tw = getvalue('event-window',varout);
tw = ceil((tw/2/1000) * sf);%put our time window into samples and half it.


%find the events in question
[sb,se] = findEvent(result,ev);
if (isempty(sb))
  temp = zeros(8,size(result,2));
  temp(1:3,:) = result(1:3,:);
  result = temp;
  disp('No such event');
  return;
end
%done marking the area
data = result;
sz = size(data);
  
%to be used for velocity peak detection
ff = sacfilt/sf;%create the filter for 20Hz (Normalized freq)
[b a] = butter(4,ff,'low');
data(1,:) =  filtfilt(b,a,data(1,:) ./ ppd(1));
data(2,:) =  filtfilt(b,a,data(2,:) ./ ppd(2));

%compute velocity in deg/Second
%take the difference of successive values
temp = [data(1:2, 2:sz(2)) [0;0]]; diff = data(1:2,:) - temp;
% compute velocity in deg per sec:
vel = (sqrt(diff(1, :).^2 + diff(2, :).^2));
vel = vel .* sf;
% cleanup the start/end:
vel(1:length(ff)) = 0; 
vel(length(vel)-length(ff)+1:length(vel))= 0;


%loop through all our events
for (kk = 1:length(sb))
  
  %get some some windows around the event in question
  tlo = tw;
  th = tw;
  if (se(kk) - tw < 1 )
    tlo =  se(kk)  - 1;
  end
  if ((se(kk) + tw) > size(result,2))
    th = size(result,2) - se(kk);
  end
  
  tblo = tw;
  tbh = tw;
  if (sb(kk) - tw < 1 )
    tblo =  sb(kk)  - 1;
  end
  if ((sb(kk) + tw) > size(result,2))
    tbh = size(result,2) - sb(kk);
  end
  
  ebx(kk) = median(result(1,sb(kk)-tblo:sb(kk)+tbh)); %begx
  eby(kk) = median(result(2,sb(kk)-tblo:sb(kk)+tbh)); %begy
  eex(kk) = median(result(1,se(kk)-tlo:se(kk)+th));  % targetx
  eey(kk) = median(result(2,se(kk)-tlo:se(kk)+th));  % targety
  
  %compute amplitude  
  amp = sqrt( sum( ([eex(kk)./ppd(1) eey(kk)./ppd(2)] ...
                     - [ebx(kk)./ppd(1) eby(kk)./ppd(2)]).^2 ) );

  if (kk ~= length(se))
    edur = sb(kk+1) - se(kk);
  else
    edur = length(result) - se(kk);
  end
  edur = edur * 1000/sf;


  %find the peak value in the interval
  pvel = max(vel(sb(kk):se(kk)));
  
  result(4, sb(kk)) = eex(kk); %target x
  result(5, sb(kk)) = eey(kk); %target y
  
  result(6, sb(kk)) = amp; %amp
  result(7, sb(kk)) = round(edur);    %interval till next event
  result(8, sb(kk)) = pvel;    %peak velocity
end


