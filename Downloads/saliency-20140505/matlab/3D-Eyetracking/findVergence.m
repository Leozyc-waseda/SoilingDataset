function out = findVergence(Rp1,Rp2,R1,R2)

%% this function encapsulates our model for 3D gaze estimation given the
% location of two dots in the space and their projections
% parameters :
% RP1 projection of point 1 (e.g. left eye) on the screen
% RP2 projection of point 2 (e.g. right eye) on the screen
% R1 a point in 3D space (e.g. location of left eye)
% R2 a point in 3D space (e.g. location of rigth eye)
% out a point in the space associated to closest distance between
% connecting lines

A = zeros(2,2);
B = zeros(2,1);
if length(Rp1)<3
Rp1 = [Rp1,0];
end
if length(Rp2) < 3 ;
Rp2 = [Rp2,0];
end ;
A(1,1) = (R1 - Rp1)*( R1 - Rp1)';
A(1,2) = -(R2 - Rp2)*(R1-Rp1)' ;
A(2,1) = - A(1,2) ;
A(2,2) = -(R2-Rp2)*(R2-Rp2)';

B(1,1) = -(Rp1-Rp2)*(R1-Rp1)';
B(2,1) = -(Rp1-Rp2)*(R2-Rp2)';

T = linsolve(A,B);

r1 = Rp1 + T(1)*(R1 - Rp1) ;
r2 = Rp2 + T(2)*(R2 - Rp2) ;

out = (r1+r2)/2 ;
