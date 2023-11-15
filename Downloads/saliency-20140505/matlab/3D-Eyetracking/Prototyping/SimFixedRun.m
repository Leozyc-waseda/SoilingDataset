%clear all;
function [Lx,Ly,Lz,Rx,Ry,Rz] = SimFixedRun(tests)
%function SimFixedRun(tests)
% Pre-load all materials we need
close all;
addpath('/lab/dicky/saliency/matlab/3D-Eyetracking/')
addpath('/lab/dicky/saliency/matlab/3D-Eyetracking/Prototyping/')

% All numbers in here are in cm

% Constants
% Assuming that the eye is in the center of the screen
%   with range between eye 6cm = 2d
d = 3;
%Er = [25, 47.6, 100];
%El = [25, 41.6, 100];
Ec = [25, 45, 100];
ScreenSize = [50, 90, 0];
LEDsize = [9.64, 14.95, 0];



ErrorRate=2;%cm. Last time we measure 20 pix per cm. We use 40pix error

% Variable that we want to test
%degree = [0:1:90];
%randXUnit = (rand(tests,1) * 2 * ErrorRate)-ErrorRate;
%randYUnit = (rand(tests,1) * 2 * ErrorRate)-ErrorRate;
%randUnit = [randXUnit , randYUnit];
%depth = [0:1:75];

%sind(degree);

for ypos = [0:1:90],
    %depth
    R1=[ScreenSize(1)/2-LEDsize(1)/2,ypos];
    
    [aels{ypos+1},aers{ypos+1}] = fixDepthModelSimulation_ret(Ec,d,R1,LEDsize(1),LEDsize(2),ErrorRate,ErrorRate,tests);
end

%size(aels{1})
%return;

%aelz = aels;
%aerz = aers;

% Pre-alloc
[x,y] = meshgrid(0:90, 1:75);
Lx = x;
Ly = x;
Lz = x;
Rx = x;
Ry = x;
Rz = x;
MinL = [-1 -1 1000;-1 -1 1000;-1 -1 1000];
MinR = [-1 -1 1000;-1 -1 1000;-1 -1 1000];
for i = 1:91,
    for j = 1:75,
        Lx(j,i) = aels{i}(j,1)/tests;
        Ly(j,i) = aels{i}(j,2)/tests;
        Lz(j,i) = aels{i}(j,3)/tests;
        Rx(j,i) = aels{i}(j,1)/tests;
        Ry(j,i) = aels{i}(j,2)/tests;
        Rz(j,i) = aels{i}(j,3)/tests;
        if (MinL(1,3) > Lx(j,i))
            MinL(1,:) = [ i-1 j Lx(j,i)];
        end;
        if (MinL(2,3) > Ly(j,i))
            MinL(2,:) = [ i-1 j Ly(j,i)];
        end;
        if (MinL(3,3) > Lz(j,i))
            MinL(3,:) = [ i-1 j Lz(j,i)];
        end;
        
        if (MinR(1,3) > Rx(j,i))
            MinR(1,:) = [ i-1 j Rx(j,i)];
        end;
        if (MinR(2,3) > Ry(j,i))
            MinR(2,:) = [ i-1 j Ry(j,i)];
        end;
        if (MinR(3,3) > Rz(j,i))
            MinR(3,:) = [ i-1 j Rz(j,i)];
        end;
    end
end

fprintf('Left Eye X: min=%f Y=%d Z=%d\n',MinL(1,3), MinL(1,1), MinL(1,2));
fprintf('Left Eye Y: min=%f Y=%d Z=%d\n',MinL(2,3), MinL(2,1), MinL(2,2));
fprintf('Left Eye Z: min=%f Y=%d Z=%d\n',MinL(3,3), MinL(3,1), MinL(3,2));

fprintf('Right Eye X: min=%f Y=%d Z=%d\n',MinR(1,3), MinR(1,1), MinR(1,2));
fprintf('Right Eye Y: min=%f Y=%d Z=%d\n',MinR(2,3), MinR(2,1), MinR(2,2));
fprintf('Right Eye Z: min=%f Y=%d Z=%d\n',MinR(3,3), MinR(3,1), MinR(3,2));


figure;
z = Lx;
maxval = max(max(z));
minval = min(min(z));
%fprintf('Left Eye X : min=%f\n', minval)
zz = -z;
surf(x,y,zz)
%axis([0 90 1 75 -maxval/10000 -minval -maxval/10000 -minval])
hold on;

figure;
z = Ly;
maxval = max(max(z));
minval = min(min(z));
%fprintf('Left Eye Y : min=%f\n', minval)
zz = -z;
surf(x,y,zz)
%axis([0 90 1 75 -maxval/10000 -minval -maxval/10000 -minval])
hold on;

figure;
z = Lz;
maxval = max(max(z));
minval = min(min(z));
%fprintf('Left Eye Z : min=%f\n', minval)
zz = -z;
surf(x,y,zz)
%axis([0 90 1 75 -maxval/10000 -minval -maxval/10000 -minval])
hold on;

figure;
z = Rx;
maxval = max(max(z));
minval = min(min(z));
%fprintf('Right Eye X : min=%f\n', minval)
zz = -z;
surf(x,y,zz)
%axis([0 90 1 75 -maxval/10000 -minval -maxval/10000 -minval])
hold on;

figure;
z = Ry;
maxval = max(max(z));
minval = min(min(z));
%fprintf('Right Eye Y : min=%f\n', minval)
zz = -z;
surf(x,y,zz)
%axis([0 90 1 75 -maxval/10000 -minval -maxval/10000 -minval])
hold on;

figure;
z = Rz;
maxval = max(max(z));
minval = min(min(z));
%fprintf('Right Eye Z : min=%f\n', minval)
zz = -z;
surf(x,y,zz)
%axis([0 90 1 75 -maxval/10000 -minval -maxval/10000 -minval])
hold on;



%[pl,pr] = geModel(d,ec,R1,teta,a,b)
%[le,re] = locateEyes(pl,pr,k)