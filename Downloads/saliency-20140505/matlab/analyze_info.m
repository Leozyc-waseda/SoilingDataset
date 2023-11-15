function analyze_info(data, avgi, mini, maxi, pstring)
%function analyze_info(data, avgi, mini, maxi[, pstring])
%
% makes a nice plot of info results;
% data contains one line per image, and successive columns are
% relative info at corresponding attended locations (each line
% is padded with NaN after 700ms expired).
% pstring gives the plot string

if nargin < 5, pstring = 'w'; end;
[nr nc] = size(data);

%subplot(1,3,1); plot(data');
%xlabel('Attended location number');
%ylabel('Local information');
%title(['Original Data (n=',int2str(nr),')'])
avg1 = zeros(1, nc); stdev1 = zeros(1, nc); ns = zeros(1, nc);
avg2 = zeros(1, nc); stdev2 = zeros(1, nc);
for ii = 1:nc
 for jj = 1:nr
  if finite(data(jj, ii))
   avg1(ii) = avg1(ii) + data(jj, ii) / avgi(jj);
   stdev1(ii) = stdev1(ii) + (data(jj, ii) / avgi(jj))^2;
   avg2(ii) = avg2(ii) + data(jj, ii) / maxi(jj);
   stdev2(ii) = stdev2(ii) + (data(jj, ii) / maxi(jj))^2;
   ns(ii) = ns(ii) + 1;
  end
 end
 if ns(ii) > 1
  avg1(ii) = avg1(ii) / ns(ii);
  stdev1(ii) = sqrt(ns(ii)/(ns(ii)-1)*(stdev1(ii)/ns(ii)-avg1(ii)^2));
  avg2(ii) = avg2(ii) / ns(ii);
  stdev2(ii) = sqrt(ns(ii)/(ns(ii)-1)*(stdev2(ii)/ns(ii)-avg2(ii)^2));
 else
  avg1(ii) = NaN;
  stdev1(ii) = NaN;
  avg2(ii) = NaN;
  stdev2(ii) = NaN;
 end
end

%subplot(1,3,2);
errorbar(1:length(avg1), avg1, stdev1/sqrt(nr), stdev1/sqrt(nr), pstring);
hold on;
plot([1 length(avg1)], [1 1], 'b');
xlabel('Attended location number');
ylabel('Local SFC / avg or max SFC');
title([ 'Average and stderr (n=',int2str(nr),')' ])
%subplot(1,3,3);
errorbar(1:length(avg2), avg2, stdev2/sqrt(nr), stdev2/sqrt(nr), pstring);
hold on;
plot([1 length(avg2)], [1 1], 'b');
%xlabel('Attended location number');
%ylabel('Local information / max information');
%title([ 'Average and stderr (n=',int2str(nr),')' ])
