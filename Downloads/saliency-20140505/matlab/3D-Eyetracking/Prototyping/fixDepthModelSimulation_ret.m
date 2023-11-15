function [aels,aers] = fixDepthModelSimulation_ret(ec,d,R1,a,b,ex,ey,ns)

%%  function fixDepthModelSimulation(ec,d,R1,a,b,ex,ey,ns)
%This function runs a simulation for tracback calibration paradigm
% ec a vector associated to midpoint of eyes
% d distance from each of eyes to midpoint
% R1 two dimensional vector associate to LED #1
% a distance between LED #1 and LED #2
% b distance between LED #1 and LED #4
% ns number of samples

aels=zeros(ec(3)-25,3);%this is averages of left eye locations, each row shows the average of eye positions for a specific depth
aers=zeros(ec(3)-25,3);%this is averages of right eye locations
rel = [ec(1) ec(2)-d ec(3)];
rer = [ec(1) ec(2)+d ec(3)];
for ss = 1:ns
els=[];
ers=[];
    for h=1:ec(3)-25
        [pl,pr] = geoModel(d,ec,[R1,h],0,a,b,ex,ey);
        [el,er]=trackbackEyes(pl,pr,[R1,h],a,b);
        els=[els;abs(el-rel)]; 
        ers = [ers;abs(er-rer)];
    end
aels = aels + abs(els) ;
aers = aers + abs(ers) ;
end
