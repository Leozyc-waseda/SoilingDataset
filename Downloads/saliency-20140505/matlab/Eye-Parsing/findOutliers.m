function [idx,d] = findOutliers(sc, ey, dist)
%function idx = findOutliers(sc, ey, dist)
% returns indices of NON-outliers, i.e., where the distance between ey and sc
% is smaller than dist

diff = sc - ey;
d = sqrt(diff(1,:).^2 + diff(2,:).^2);
idx = find(d < dist);
