function sc = calibAffine(ey, M)
%function sc = calibAffine(ey, M)
% This function performs the affine transformation M on points ey.

sz = size(ey);
e = [ey(1,:); ey(2,:); ones(1,sz(2))];
s = M * e;
sc = s(1:2, :);
