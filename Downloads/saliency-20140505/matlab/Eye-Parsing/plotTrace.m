function plotTrace(data, s, c, nfr)
%function plotTrace(data, s, c, nfr)
% plot a data trace using style 's'; clear previous plots if 'c'
% given; use a fixed display range unless nfr is given and non-zero
% data is given as a 3xN matrix, columns are in (x,y,pd) format
% if nfr is 1, calling function is responsible for setting range
if (nargin >= 4 & nfr ~= 0), useFixedRange = 1;
else useFixedRange = 0; end
if (nargin <= 1), s = '-'; end
sz = size(data);

subplot(4, 1, 1); if (nargin >= 3 & c ~= 0), hold off; end
plot(data(1, :), data(2, :), s); ylabel('Pupil X/Y');
grid on; hold on;

subplot(4, 1, 2); if (nargin >= 3 & c ~= 0), hold off; end
plot(data(1, :), s); ylabel('Pupil X');
if (useFixedRange ~= 0), axis([ 1 sz(2) 0 512]); end
set(gca, 'XTickLabel', []); grid on; hold on;

subplot(4, 1, 3); if (nargin >= 3 & c ~= 0), hold off; end
plot(data(2, :), s); ylabel('Pupil Y');
if (useFixedRange ~= 0), axis([ 1 sz(2) 0 512]); end
set(gca, 'XTickLabel', []); grid on; hold on;

if (sz(1) > 2)
  subplot(4, 1, 4); if (nargin >= 3 & c ~= 0), hold off; end
  plot(data(3, :), s); ylabel('Pupil Diam'); grid on; hold on;
  if (useFixedRange ~= 0), axis([ 1 sz(2) 0 800]); end
end
