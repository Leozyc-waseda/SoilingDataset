function [amp,pvel] = mainsequence(glob,varargin)

%returns the main sequence of some data
varout = parseinputs(varargin);
sf = getvalue('Sf',varout);
ppd = getvalue('ppd',varout);%for amp and velocity conversions
sacfilt = getvalue('sac-filter',varout)
scsz = getvalue('screen-size',varout)

[filez,glob] = strip_file_path(glob);
amp = [];
pvel = [];

for (ii = 1:length(filez))
  data = [];
  temp = [];
  vel = [];
  diff = [];
  
  fnam = [glob,filez{ii}];
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
  [sb,se] = findEvent(data,1,1);%find saccades not including
                                %combined and junk

  
  %f = find( (data(1,sb) < 0) | (data(1,sb) > scsz(1)) |...
  %          (data(2,sb) < 0) | (data(2,sb) > scsz(2)) |...
  %          (data(1,se) < 0) | (data(1,se) > scsz(1)) |...
  %          (data(2,se) < 0) | (data(2,se) > scsz(2)) );
  %
  
  f = find( (sb == 1) | (se == length(data)) );
  sb(f) = [];
  se(f) = [];
  f = [];
  
  for (jj = 1:length(sb))
      if ((sb(jj)-1 > 0) && (se(jj)+1 < length(data)))
          if (data(3,sb(jj)-1) == 6) || (data(3,se(jj)+1) == 6)
              f = [f,jj];
              error('You should parse your data for no combines saccades first');
          end
      end
  end
  sb(f) = [];
  se(f) = [];

  eb = datao(1:2,sb);
  ee = datao(1:2,se);

  eb(1,:) = eb(1,:)./ppd(1);  ee(1,:) = ee(1,:)./ppd(1);
  eb(2,:) = eb(2,:)./ppd(2);  ee(2,:) = ee(2,:)./ppd(2);
  amp = [amp,sqrt( sum( (ee - eb).^2 ) )];
  
  f = find (amp < 2);
% [f,ff] = findevent(data,6,1);
   if (~isempty(f))
       error('We shouldnt have saccades less than 2 degrees!');
   end
  
  %find the peak valuein the interval
  for (kk = 1:length(sb))
    pvel = [pvel, max(vel(sb(kk):se(kk)))];
  end
  ii
end


