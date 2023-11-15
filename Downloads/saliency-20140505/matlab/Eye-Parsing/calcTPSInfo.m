function [tpsInfo] = calcTPSInfo(landmarks)
%function [tpsInfo] = calcTPSInfo(landmarks)
%Given a set of landmarks (Mx4 where Mx2 is target, next Mx2 is template)
%it calculates the necessary information to fit a thin-plate spline over the
%data
%tpsInfo has the form
%tpsInfo.points     The XY location of all of the target points (Mx2)
%tpsInfo.a          A 3x1 vector of the affine transformation info.
%tpsInfo.w          A Mx1 vector containing the weights for each of the points
%
% Copyright (C) 2002 John F. Meinel Jr. <jfmeinel@engineering.uiowa.edu>
% Last Modified: $Date: 2003/05/27 03:35:48 $
% $Revision: 1.1.1.1 $

[numPts] = size(landmarks,1);%The number of rows is the number of landmarks

%First thing is to calculate the U matrix of each point to each other point
x = landmarks(:,1);
y = landmarks(:,2);

%Fill a matrix with the x location, and a matrix with the y locations
Px = ones(numPts, 1) * x';%This contains the x value at each location
Px = Px - Px';%Find the distance from every x to every other
%Do the same for y
Py = ones(numPts, 1) * y';
Py = Py - Py';

%Now the distance is the Eulerian distance, but we only care about the squared diff
r_sq = Px.^2 + Py.^2;

%The U matrix is r^2*log(r^2)
locs = find(r_sq > 0);%You don't want to take the log of zero
%And 0^2 log(0^2) = asymptotically approaches 0
U = zeros(size(r_sq));
U(locs) = r_sq(locs) .* log(r_sq(locs));

%P is [1 x1 y1; 1 x2 y2] ...
P = [ones(numPts, 1) x y];
L = zeros(numPts + 3, numPts + 3);
L(1:numPts, 1:numPts) = U;
L(1:numPts,numPts+1:numPts+3) = P;
L(numPts+1:numPts+3,1:numPts) = P';

%Now determine the V vector, which is just the original landmarks, augmented by 3-0's
V = zeros(numPts+3, 2);
V(1:numPts,:) = landmarks(:,3:4);

%Now to determine the weights, just take the inverse
W = (L^-1)*V;

tpsInfo.points = landmarks(:,1:2);
tpsInfo.w = W(1:numPts, :);
tpsInfo.a = W(numPts+1:numPts+3,:);
