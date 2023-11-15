function [le,re] = FDdoAllInOne(Psych,Eye)
%% [le,re] = FDdoAllInOne(Psych,Eye)
%%this function takes in PSYCH and EYELINK data structures wich has
%% off-screen calibration data and spits out the location of left and right
%% eye

a=10;% distance between LED#1 and LED #2
b=15;%distance between LED#1 and LED # 4
ppcx = 21.51 ;%this is pixel per centimeter along x axis (vertical)
ppcy = 21.6 ; %this is pixel per centimeter along y axis (horizontal)
R1=[14,51,68]; % we assume this is the postion of LED 1 

[pl,pr] = inspectSessions(Psych,Eye);
pl(1,:) = pl(1,:)/ppcx;
pl(2,:) = pl(2,:)/ppcy;
pr(1,:) = pr(1,:)/ppcx;
pr(2,:) = pr(2,:)/ppcy ;



[le , re] = trackbackEyes(pl,pr,R1,a,b);