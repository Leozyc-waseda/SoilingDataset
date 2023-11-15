function [defPts] = deformPoints(pts, tpsInfo, targetPts, fraction)
%function [defPts] = deformPoints(pts, tpsInfo, targetPts, fraction)
%This function deforms a set of points (Mx2 matrix)
%using a thin-plate spline function
%tpsInfo contains the necessary information for using a thin plate spline
%f(x,y) = a1+a2*x+a3*y+sum(w_i U(p_i-(x,y)));
%See calcTPSInfo for information on tpsInfo
%targetPts is optional, if not supplied, this function uses the points
%specified by tpsInfo
%if fraction is supplied, then it won't do the full distortion, it will
%only distort a fraction of the total distance.
%
% Copyright (C) 2002 John F. Meinel Jr. <jfmeinel@engineering.uiowa.edu>
% Last Modified: $Date: 2003/05/27 03:35:48 $
% $Revision: 1.1.1.1 $

numPts = size(pts, 1);
P = [ones(numPts, 1) pts];%P has columns: 1, x, y with M rows

if(exist('targetPts') ~= 1 | isempty(targetPts))
    targetPts = tpsInfo.points;
end
numLandmarks = size(targetPts, 1);
Px = ones(numPts, 1) * targetPts(:,1)';%This contains the x value at each location
%Find the distance from every landmark to every point
X = pts(:,1) * ones(numLandmarks, 1)';
Px = Px - X;
clear X;

Py = ones(numPts, 1) * targetPts(:,2)';%This contains the y value at each location
%Find the distance from every landmark to every point
Y = pts(:,2) * ones(numLandmarks, 1)';
Py = Py - Y;
clear targetPts Y;

%Now calculate r^2
r_sq = Px.^2 + Py.^2;
clear Px Py;

%The U matrix is r^2*log(r^2)
locs = find(r_sq > 0);%You don't want to take the log of zero
%And 0^2 log(0^2) = asymptotically approaches 0
U = zeros(size(r_sq));
U(locs) = r_sq(locs) .* log(r_sq(locs));
clear r_sq;

%defPts should be Mx2, the first part is the affine portion
%the second part is the thin-plate spline portion
defPts = (tpsInfo.a' * P')' + (tpsInfo.w'*U')';
clear P U;

if(exist('fraction')==1)
    defPts = ((defPts - pts) * fraction) + pts;
end
clear pts;