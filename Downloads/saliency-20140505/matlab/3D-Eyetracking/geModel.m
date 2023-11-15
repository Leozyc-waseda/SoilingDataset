function [pl,pr] =geModel(d,ec,R1,teta,a,b)

%
%this function is for generating data for testing our 3D calibration
%model, it takes physical configurations of eyes and LED panel and
%generates data points that can be measured with binocular EyeLink gaze
%tracking.
% prarmeters:
% d : half of distance between two eyes 
% ec : a vector corresponding to coordinates of bisector of eye centers
% R1 : a vector corresponding to coordinates of LED #1 (top-right one)
% teta : angle between LED plane and presentation screen
% a : distance between LED # 1 and LED # 2 or LED # 4 and LED #4
% b : distance between LED # 1 and LED # 4 or LED # 3 and LED #4
%
% pl: a 2x4 matrix, the first row related to x values and the second rows
% related to y values and ith column related to projection of left eye on
% the screen relative to LED # i
% pr: a 2x4 matrix, the first row related to x values and the second rows
% related to y values and ith column related to projection of right eye on
% the screen relative to LED # i
%
% pl and pr along with ecy (or any estimation or ecy) can be fed to
% locateEyes.m


% let's find el and er, vectors associated to left eye and right eye
el = ec;
er = ec ;
el(2) = ec(2)-d ;
er(2) = ec(2) + d;

% now let's find R2, R3 and R4
R2=R1 ;
R2(1) = R2(1)+a;
R3=R2;
R3(2)=R2(2) - b* cos(teta); R3(3) = R2(3)+ b*sin(teta);
R4 = R3 ;
R4(1) = R3(1)-a ;

%these parameters encapsulate the intersection of eye-Ri ray and
%presentation screen
t1= ec(3)/(ec(3) - R1(3));
t2 = t1 ;
t3 = ec(3)/(ec(3) - R3(3));
t4=t3 ;

% now that we know the parameter we can actually find the intersection
% locations

% for the left eye 
pl1= el + t1*(R1 - el);
pl2= el + t2*(R2 - el);
pl3= el + t3*(R3 - el);
pl4= el + t4*(R4 - el);
% for the right eye
pr1= er + t1*(R1 - er);
pr2= er + t2*(R2 - er);
pr3= er + t3*(R3 - er);
pr4= er + t4*(R4 - er);

 % and now reformatting the our data into our favorit form
pl = [pl1(1) pl2(1) pl3(1) pl4(1); pl1(2) pl2(2) pl3(2) pl4(2)];
pr = [pr1(1) pr2(1) pr3(1) pr4(1); pr1(2) pr2(2) pr3(2) pr4(2)];

