function [le,re] = locateEyes(pl,pr,k,a ,b)
% this function is the key function that solves our calibration geometrical
% model for finding location of both eyes. this function takes in
% projection of left eye and projection of right eye on presentation screen
% relative to LEDs and returs two vectors corresponding to eye locations
% assumptions : 1. All four LEDs are on a plane arranged on the corners of a
% rectangle 2. those sides of above mentioned rectangle that connect LED #
% and LED # 2, and LED #3 and LED #4 are parallet to presentation screen 3.
% the lines connecting two eyes are parallel to presentation screen.
% prameters :
% pl : a 2x4 matrix with rows corresponding to x and y coordinate of the
% projection dots (for left eye) and column i corresponding to LED # i
%pr : a 2x4 matrix with rows corresponding to x and y coordinate of the
% projection dots (for right eye) and column i corresponding to LED # i
% k : y coordinate of bisector of two eyes
% le : vector corresponding to location of left eye
% re : vector corresponding to location or right ey

% a is the distance between LED #1 and LED #2 in cm
% b is the distance between LED #1 and LED #4 in cm

if nargin < 5 || isempty(a) 
    a = 10 ;
end

if nargin < 5 || isempty(b)
    b = 15;
end

lx1 = pl(1,1);
lx2 = pl(1,2);
lx3 = pl(1,3);
lx4 = pl(1,4) ;
ly1 = pl(2,1);
ly2 = pl(2,2);
ly3 = pl(2,3);
ly4 = pl(2,4) ;

rx1 = pr(1,1);
rx2 = pr(1,2);
rx3 = pr(1,3);
rx4 = pr(1,4) ;
ry1 = pr(2,1);
ry2 = pr(2,2);
ry3 = pr(2,3);
ry4 = pr(2,4) ;

%d=(a*(-ly1 + ry1))/(2*(a + rx1 - rx2));
d=(a*(-ly3 + ry3))/(2*(a - rx3 + rx4));
ecx=(rx1*rx3 - rx2*rx4)/(rx1 - rx2 + rx3 - rx4);
ecz= sqrt(4*b^2*(rx1 - rx2)^2*(rx3 - rx4)^2 - a^2*(2*k*(-rx1 + rx2 - rx3 + rx4) + (rx3 - rx4)*(ly1 + ry1) +(rx1 - rx2)*(ly3 + ry3))^2)/(2*a*sqrt((rx1 - rx2 + rx3 - rx4)^2));
le=[ecx,k-d,ecz];
re=[ecx,k+d,ecz];