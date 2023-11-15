function error= FDEyetrackingSim(ec,d,R1,a,b,ex,ey,h,ns)
%%  FDEyetrackingSim(ec,d,R1,a,b,ex,ey,h,ns)
% this function runs a simulation of eye tracking on a 3D grid in the space
% given that the initial calibration is done using Fixed Depth Calibration
% Method
% arguments:
% ec ground truth center of the eye
% d ground truth half distance between two eyes
% R1 location of LED#1 (two dimensional vector please!)
% a distance between LED#1 and LED#2
% b distance between LED#1 and LED#4
% ex error of reading eye position along x direction
% ey error of reading eye position along y direction
% h distance between LED plate and presentation screen
% number of samples


rel = [ec(1) ec(2)-d ec(3)];
rer = [ec(1) ec(2)+d ec(3)];


%cec=(cle+cre)/2;
x1=10;xs=10;x2=50;
y1=10;ys=10;y2=80;
z1=-80;zs=20;z2=80;
error = zeros((x2-x1+xs)/xs,(y2-y1+ys)/ys,(z2-z1+zs)/zs,3);

for ss = 1:ns
    [pl,pr] = geoModel(d,ec,[R1,h],0,a,b,ex,ey);
    [cle,cre]=trackbackEyes(pl,pr,[R1,h],a,b);
for xx=x1:xs:x2
    for yy=y1:ys:y2
        for zz=z1:zs:z2
            
            %let's first see where are the left and right gaze
            [pl,pr] = geoModel_3D_dot(rel,rer,[xx,yy,zz],ex,ex);
            % now let's with our calibration data where is the best
            % suggested location for the recorded gaze locations
            ver = findVergence(pl',pr',cle,cre);
            error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,1)= error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,1) + abs(xx-ver(1));
            error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,2)= error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,2)+ abs(yy - ver(2));
            error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,3)= error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,3)+ abs(zz - ver(3));
        end
    end
end
end
error=error/ns ;

%[sx sy sz] = sphere;
sx=[0,0,0,0,0;-.5,0,.5,0,-.5;-1,0,1,0,-1;-.5,0,.5,0,-.5;0,0,0,0,0];
sy=[0,0,0,0,0;0,-.5,0,.5,0;0,-1,0,1,0;0,-.5,0,.5,0;0,0,0,0,0];
sz=[-1,-1,-1,-1,-1;-.5,-.5,-.5,-.5,-.5;0,0,0,0,0;.5,.5,.5,.5,.5;1,1,1,1,1];


figure;
hold on
for xx=x1:xs:x2
    for yy=y1:ys:y2
        for zz=z1:zs:z2
            erx = error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,1);
            ery = error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,2);
            erz = error((xx-x1+xs)/xs,(yy-y1+ys)/ys,(zz-z1+zs)/zs,3);
            %drawErrorCross([xx yy zz] , [er er er]);
            surf(sx*erx+xx , sy*ery+yy , sz*erz+zz);
            
            % For artistic approach uncomment below and comment out above
            %surf(sx*erx+ xx , sy*ery +yy , sz*erz+ zz,'FaceColor', [xx/50.0 yy/80.0 (zz+80)/160.0],'Edgecolor','none');
            %alpha(0.6);
        end
    end
end
daspect([1,1,1]);
hold off


