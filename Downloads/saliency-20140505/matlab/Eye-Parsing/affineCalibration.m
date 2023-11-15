function M = affineCalibration(sc, ey)
%function M = affineCalibration(sc, ey)
% Compute affine matrix so that M * [xeye; yeye; 1] approximates
% [xscr; yscr; 1]

sz = size(ey);
e = [ey(1,:); ey(2,:); ones(1,sz(2))];
s = [sc(1,:); sc(2,:); ones(1,sz(2))];

% do a least-square solution to M*e = s
M = s / e;
