function result = modelfreestats(glob,var)
%actually does the model free work, call modelfree(path) to
%instantiate this function. This will ignore combined saccades and
%saccades with tracker errors. 

varout = parseinputs(var);
sf = getvalue('Sf',varout);
ppd = getvalue('ppd',varout);%for amp and velocity conversions
sacfilt = getvalue('sac-filter',varout);
scsz = getvalue('screen-size',varout);
strict = getvalue('modelfree-strict',varout);
stats = getvalue('stats',varout);

[filez,glob] = strip_file_path(glob);

result = [];

for (ii = 1:length(filez))
  amp=[];
  pvel=[];
  angle=[];
  intersac=[];
  bdur=[];
  adur=[];
  
  data = [];
  temp = [];
  vel = [];
  diff = [];
  
  fnam = [glob=[];filez{ii}];
  data = loadCalibTxt(fnam); %load the file
  datao = data;

  %removeShort(data,varargin);
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

  [sb,se] = findEvent(data,6,1);%see if combined saccades
  if (~isempty(sb))
    disp(['The data contains combined saccades (markup 6). Please ', ...
          'consider re-marking with pro-anglethresh=0 to prevent ', ...
          'saccade combination for model-free statistics. If ', ...
          'modelfree-strict=0 (default) type 6 saccades will be ignored, ',...
          'increasing the intersaccadic interval estimate of the ',... 
          'previous saccade. However, if modelfree-strict=1 then ', ...
          'the previous saccades intersaccadic interval estimate will be marked as 0']);
  end
  
    [sb,se] = findEvent(data,3,1);%see if saccades in blink
  if (~isempty(sb))
    disp(['The data contains saccades in blink(markup 3). This marking ', ...
          'indicates that the eye tracker lost signal for a moment, but ', ...
          'the distance traveled when signal was regained indicated a saccade.',...
          'This marking may also occur with no saccade, if the eye tracker pupil and corneal', ...
          'reflection isolation is poor and changes rapidly. If the setting',...
          'modelfree-strict=0 (default) type 3 saccades will be ignored, ',...
          'increasing the intersaccadic interval estimate of the ',... 
          'previous saccade. However, if modelfree-strict=1 then ', ...
          'the previous saccades intersaccadic interval estimate will be marked as 0']);
  end
 
  
  [sb,se] = findEvent(data,1);%find saccades
  sacbad = find( (datao(3,sb) == 6) || (datao(3,sb) == 3) );
  
  f = find( (sb == 1) | (se == length(data)) );
  sb(f) = [];
  se(f) = [];
  f = [];
  
  eb = datao(1:2,sb);
  ee = datao(1:2,se);

  %if flipped in file name turn it around
  if (regexp(glob,'FLIPPED'))
    eb(1,:) = scsz(1) - eb(1,:);eb(2,:) = scsz(2) - eb(2,:);
    ee(1,:) = scsz(1) - ee(1,:);ee(2,:) = scsz(2) - ee(2,:);
  end

  %amplitude and angle calculation
  eb(1,:) = eb(1,:)./ppd(1);  ee(1,:) = ee(1,:)./ppd(1);
  eb(2,:) = eb(2,:)./ppd(2);  ee(2,:) = ee(2,:)./ppd(2);
  amp = sqrt( sum( (ee - eb).^2 ) );
  angle = -atan2(ee(2,:) - eb(2,:),ee(1,:) - eb(1,:)) .* 180.0 ./ pi;  
  tres(:,strkey('amplitude',stats)) = amp';
  tres(:,strkey('angle',stats)) = angle';
  
  %lets get all our fixations
  [fixbeg,fixend] = findevent(data,0);
    
  %loop through the saccades and calculate a few things
  disp([num2str(length(sb)),':: Saccades']);
  for (kk = 1:length(sb))
    
    %find the peak value in the interval
    pvel = [pvel, max(vel(sb(kk):se(kk)))];
    
    %now the intersaccadic interval
    if (kk ~= length(sb))
      if (isempty(find(datao(3,se(kk):sb(kk+1)) == 3)) && ...
          isempty(find(datao(3,se(kk):sb(kk+1)) == 6)) || (strict == 0))
        tdat = (sb(kk+1) - se(kk)) * 1000/sf;
      else
        tdat = 0;
      end
    else
      tdat = 0;
    end
    intersac = [intersac,tdat];
    
    %now if see if there was a fixation before
    %and if so store its duration
    f = find(sb(kk)-1 == fixend);
    if (length(f) > 1)
      error('Two fixations with the same endpoint in time? FATAL ERROR');
    end
    if (isempty(f))
      tdat = 0;
    else
      if ((fixbeg(f) == 1) || (fixend(f) == length(data)))
        tdat = 0;
      else
        tdat = (fixend(f) - fixbeg(f)) * 1000/sf;      
      end
    end
    bdur = [bdur,tdat]; 
       
    %now if see if there was a fixation before
    %and if so store its time
    f = find(se(kk)+1 == fixbeg);
    if (length(f) > 1)
      error('Two fixations with the same endpoint in time? FATAL ERROR');
    end
    if (isempty(f))
      tdat = 0;
    else
      if ((fixbeg(f) == 1) || (fixend(f) == length(data)))
        tdat = 0;
      else
        tdat = (fixend(f) - fixbeg(f)) * 1000/sf;
      end
    end
    adur = [adur,tdat]; 
  
  end%end loop through sacacdes
  pvel(sacbad) = 0;
  tres(:,strkey('peakvel',stats)) = pvel';
  tres(:,strkey('sacinterval',stats)) = intersac';
  tres(:,strkey('fixafterdur',stats)) = adur';
  tres(:,strkey('fixbeforedur',stats)) = bdur';
  
  result = [result;tres];
end





