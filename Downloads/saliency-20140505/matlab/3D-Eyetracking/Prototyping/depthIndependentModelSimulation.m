function depthIndependentModelSimulation(ec,d,R1,a,b,ex,ey,ns)

%%  function depthIndependentModelSimulation(ec,d,R1,a,b,ex,ey,ns)
%This function runs a simulation for depth independent calibration paradigm
% ec a vector associated to midpoint of eyes
% d distance from each of eyes to midpoint
% R1 two dimensional vector associate to LED #1
% a distance between LED #1 and LED #2
% b distance between LED #1 and LED #4
% ns number of samples

teta=pi/180:pi/180:pi - pi/180 ;

aels=zeros(length(teta),3);%this is averages of left eye locations, each row shows the average of eye positions for a specific depth
aers=zeros(length(teta),3);%this is averages of right eye locations
rel = [ec(1) ec(2)-d ec(3)];% ground truth left eye position
rer = [ec(1) ec(2)+d ec(3)];% ground truth rigth eye position

for ss = 1:ns
els=[];
ers=[];
    for ii=1:length(teta)
        [pl,pr] = geoModel(d,ec,R1,teta(ii),a,b,ex,ey);
        [el,er] = locateEyes(pl,pr,ec(2),a,b);
        els=[els;abs(el-rel)]; 
        ers = [ers;abs(er-rer)];
    end
aels = aels + abs(els) ;
aers = aers + abs(ers) ;
end

figure;
plot(1:length(teta),aels(:,1)/ns);
hold on;
plot(1:length(teta),aers(:,1)/ns,'g');
figure
plot(1:length(teta),aels(:,2)/ns);
hold on;
plot(1:length(teta),aers(:,2)/ns,'g');
figure
plot(1:length(teta),aels(:,3)/ns);
hold on;
plot(1:length(teta),aers(:,3)/ns,'g');
