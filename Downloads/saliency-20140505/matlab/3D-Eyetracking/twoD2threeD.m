function out = twoD2threeD(in,el,er)
%% out = twoD2threeD(in,el,er) 
%% this function takes in a data structure which has left and right eye
%% locations in addition to left and right eye positions and spits our the
%% gaze location in 3D space.
out = [];
l = length(in);

%let's first swap the x and y coordinations
tmp = in ;
in(:,2) = tmp(:,3);
in(:,3) = tmp(:,2);
in(:,4) = tmp(:,5);
in(:,5) = tmp(:,4);
%k = (in(:,2)+in(:,4))/2;
%in(:,2) = k ;
%in(:,4) = k ;
in(: , 2) = in(:,2)/21.51;
in(:,4) = in(:,4)/21.51;
in(:,3)=in(:,3)/21.6;
in(:,5) = in(:,5)/21.6;
for ii = 1 : l
    out = [out;in(ii,1) findVergence([in(ii,2) in(ii,3)],[in(ii,4) in(ii,5)],el,er)];    
end