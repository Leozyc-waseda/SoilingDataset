function [pl,pr] =geoModel_3D_dot(el,er,R,ex,ey)

%
%this function uses a geometric model to project both eyes on the
%presentation screen given a point in 3D space
% prarmeters:
% el : a vector associated to the center of left eye 
% er : a vector corresponding to the center of right eye
% R : a a particular point in 3D spac
% ex: error in reading x
% ey: error in reading y
% pl: a 2x1 matrix, the first row related to x values and the second rows
% related to y values which is corresponding to the projection of left eye
% pr: a 2x1 matrix, the first row related to x values and the second rows
% related to y values which is related to projection of rigth eye
% ex and ey are linear error along x and y
%

tl = el(3)/(el(3)-R(3));
tr = er(3)/(er(3)-R(3));

plx = el(1) + tl *(R(1)-el(1));
ply = el(2) + tl *(R(2)-el(2));
prx = er(1) + tr *(R(1)-er(1));
pry = er(2) + tr *(R(2)-er(2));


 % and now reformatting the our data into our favorit form
pl = [ex*rand(1)+plx; ey*rand(1)+ply];
pr = [ex*rand(1)+prx; ey*rand(1)+pry];