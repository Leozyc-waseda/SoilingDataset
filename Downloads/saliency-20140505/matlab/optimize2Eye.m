%OPTIMIZE2EYE
%function to generate optimal weights to fit eye movement data
%% a = optimize2Eye(directory) takes as input a eyeDirectory containing eyeFile
% data and imageDirectory containing images. Here the eyeFiles are looked up and the image files have the same
% names as the eyeFiles minus the extension appendage e.g. image file
% corresponding to the eyefile 100ARRAY.e-ceyeS would be 100ARRAY.png

function a = optimize2Eye(eyeDirectory, imageDirectory)

glob = dir([eyeDirectory '/' '*e*ceyeS'])

cnt =1;
for(ii=119:1:119)%size(glob,1))
    nextEyeFile   = [eyeDirectory '/' num2str(ii) 'ARRAY.e-ceyeS'];
    nextImageFile = [imageDirectory '/' num2str(ii) 'ARRAY.png'];
    eyeMap = dlmread(nextEyeFile,' ',3,0);
    saccades = extractSaccade(eyeMap);
    eyeX(:,1) = round(saccades(:,1)/16);
    eyeY(:,1) = round(saccades(:,2)/16);
    weightVec = [1,1,1];
    %fminsearch(@getNSSembed,weightVec,optimset('MaxFunEvals',5));
end
function NSS = getNSSembed(weightVec)
unix(['ezvision --just-initial-saliency-map --vc-chans=C:' num2str(weightVec(1)) 'I:' num2str(weightVec(2)) 'O:' num2str(weightVec(3)) ' --in=' nextImageFile]);
         salMap = pfmread('VCO000000.pfm');
         m = mean(salMap(:));
         s = std(salMap(:));
         NSS = 0;
        if s>0
            NSS = (salMap(eyeY,eyeX) - m) / s
            NSS = -mean(mean(NSS))
            
        end
    end

end


function sacOnly = extractSaccade(data)
cnt =1;
sacOnly=[];
for ii=1:1:length(data)
    if((data(ii,4) == 1) && (data(ii,5)~=0 | data(ii,6)~=0 | data(ii,7)~=0))
        sacOnly(cnt,:) = data(ii,:);
        cnt = cnt +1;
    end
end
end


function sacOnly = extractSaccade(data)
cnt =1;
sacOnly=[];
for ii=1:1:length(data)
    if((data(ii,4) == 1) && (data(ii,5)~=0 | data(ii,6)~=0 | data(ii,7)~=0))
        sacOnly(cnt,:) = data(ii,:);
        cnt = cnt +1;
    end
end
end

   
   
   
        