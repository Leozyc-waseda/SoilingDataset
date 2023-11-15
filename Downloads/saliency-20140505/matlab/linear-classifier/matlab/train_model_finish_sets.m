function tdata = train_model_finish_sets(ldata,tdata,conf)

itsDataType = 0;
if isfield(conf,'trainDataUse') 
    if strcmp(conf.trainDataUse,'base')
        itsDataType = 1;
    elseif strcmp(conf.trainDataUse,'diff')
        itsDataType = 2;
    elseif strcmp(conf.trainDataUse,'target') 
        itsDataType = 3; 
    else
        error('conf.trainDataUse is not valid. Got %s',conf.trainDataUse);
    end
else
    error('conf.trainDataUse is not set');
end


itsFrames = 11;

tdata.TRAIN.samples = size(tdata.TRAIN.CLASS,1) / (itsFrames * conf.feature_num); 
tdata.TEST.samples  = size(tdata.TEST.CLASS,1)  / (itsFrames * conf.feature_num); 

%tdata.TRAIN.input   = zeros(tdata.TRAIN.samples,(itsFrames * conf.trainFeatures) * 3 - 3);
%tdata.TRAIN.target  = zeros(tdata.TRAIN.samples,1);

%tdata.TEST.input    = zeros(tdata.TEST.samples,(itsFrames * conf.trainFeatures) * 3 - 3);
%tdata.TEST.target   = zeros(tdata.TEST.samples,1);

if (isfield(conf,'trainModelEasyHard') && strcmp(conf.trainModelEasyHard,'yes')) 
    easyHard = 1;
else
    easyHard = 0;
end
    
% create training set based on final saliency
for i=1:size(conf.newFeature,1)
    thisSample = 1;
    thisFrame  = 1;
    for j=1:size(tdata.TRAIN.CLASS,1)
        if tdata.TRAIN.NEW_FEATURE(j,:) == conf.newFeature(i,:)
            if thisFrame ~= 1
                if easyHard == 0 || tdata.TRAIN.NEW_CLASS(j,:) ~= 2 || tdata.TRAIN.CLASS(j,:) == conf.midBound
                    if itsDataType == 1
                        tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.AVG(j,:);
                        tdata.TRAIN.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TRAIN.STD(j,:);
                        tdata.TRAIN.input(thisSample, thisFrame + itsFrames*2 - 3 ) = tdata.TRAIN.DIFF_SPACE(j,:); 
                    elseif itsDataType == 2
                        tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.DIFF_AVG(j,:);  
                        tdata.TRAIN.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TRAIN.DIFF_STD(j,:); 
                        tdata.TRAIN.input(thisSample, thisFrame + itsFrames*2 - 3 ) = tdata.TRAIN.DIFF_SPACE(j,:); 
                    elseif itsDataType == 3           
                        tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.DIFF_TARG_AVG(j,:);  
                        tdata.TRAIN.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TRAIN.DIFF_TARG_STD(j,:); 
                        tdata.TRAIN.input(thisSample, thisFrame + itsFrames*2 - 3 ) = tdata.TRAIN.DIFF_TARG_SPACE(j,:); 
                    else
                        error('WTF? itsDataType should be set');
                    end

                    %tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.AVG(j,:);
                    %tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.DIFF_STD(j,:);
                    
                    %tdata.TRAIN.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TRAIN.STD(j,:);
                    %tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.DIFF_SPACE(j,:);  
                    
                    %tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.MAXX(j,:);
                    %tdata.TRAIN.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TRAIN.MAXY(j,:);
                    
                    %tdata.TRAIN.input(thisSample, thisFrame               - 1 ) = tdata.TRAIN.MAXY(j,:);
                end
            end
            if thisFrame == itsFrames
                if easyHard == 0
                    tdata.TRAIN.target(thisSample,1) = tdata.TRAIN.CLASS(j,:);  
                    thisSample = thisSample + 1;
                elseif tdata.TRAIN.NEW_CLASS(j,:) ~= 2 || tdata.TRAIN.CLASS(j,:) == conf.midBound
                    tdata.TRAIN.target(thisSample,1) = tdata.TRAIN.NEW_CLASS(j,:);  
                    thisSample = thisSample + 1;
                end
                thisFrame = 1;
            else
                thisFrame = thisFrame + 1;
            end
        end
    end
    
    thisSample = 1;
    thisFrame  = 1;
    for j=1:size(tdata.TEST.CLASS,1)
        if tdata.TEST.NEW_FEATURE(j,:) == conf.newFeature(i,:)
            if thisFrame ~= 1  
                if itsDataType == 1
                    tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.AVG(j,:);
                    tdata.TEST.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TEST.STD(j,:);
                    tdata.TEST.input(thisSample, thisFrame + itsFrames*2 - 3 ) = tdata.TEST.DIFF_SPACE(j,:); 
                elseif itsDataType == 2
                    tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.DIFF_AVG(j,:);  
                    tdata.TEST.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TEST.DIFF_STD(j,:); 
                    tdata.TEST.input(thisSample, thisFrame + itsFrames*2 - 3 ) = tdata.TEST.DIFF_SPACE(j,:); 
                elseif itsDataType == 3           
                    tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.DIFF_TARG_AVG(j,:);  
                    tdata.TEST.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TEST.DIFF_TARG_STD(j,:); 
                    tdata.TEST.input(thisSample, thisFrame + itsFrames*2 - 3 ) = tdata.TEST.DIFF_TARG_SPACE(j,:); 
                else
                    error('WTF? itsDataType should be set');
                end
                
                
                %tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.AVG(j,:);
                %tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.DIFF_STD(j,:);
                
                %tdata.TEST.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TEST.STD(j,:);
                %tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.DIFF_SPACE(j,:);
                
                %tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.MAXX(j,:);
                %tdata.TEST.input(thisSample, thisFrame + itsFrames   - 2 ) = tdata.TEST.MAXY(j,:);
                
                %tdata.TEST.input(thisSample, thisFrame               - 1 ) = tdata.TEST.MAXY(j,:);
            end
            if thisFrame == itsFrames
                if easyHard == 0
                    tdata.TEST.target(thisSample,1) = tdata.TEST.CLASS(j,:);
                else
                    tdata.TEST.target(thisSample,1) = tdata.TEST.NEW_CLASS(j,:);
                end
                thisSample = thisSample + 1;
                thisFrame = 1;
            else
                thisFrame = thisFrame + 1;
            end 
        end
    end
end