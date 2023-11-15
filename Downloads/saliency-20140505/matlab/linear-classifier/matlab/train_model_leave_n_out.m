function tdata = train_model_leave_n_out(ldata,conf)

itsFrames   = conf.trainEnd - conf.trainStart + 1;
totalFrames = 11;

tdata     = struct('description','Holds new test data');
tdata.SVM = struct('description','Holder for SVM data');

% do some initial work on class sorting
feature_label = conf.feature_label;
feature_num   = conf.feature_num;

ldata.NEW_FEATURE   = zeros(size(ldata.CLASS,1),1);
ldata.feature_label = feature_label;

% match training feature lables to their id number
for i=1:size(ldata.CLASS,1)
    for j=1:feature_num
        if strcmp(ldata.FEATURE(i,1),feature_label{j})
            ldata.NEW_FEATURE(i,1) = j;
            break;
        end
    end
end

% match training feature lables to their id number
for i=1:feature_num
    for j=1:conf.trainFeatures
        if strcmp(feature_label{i},conf.featureTrain{j})
            conf.newFeature(j,:) = i;
        end
    end
end

% Take the data which is in one column and create feature sets per row
for i=1:size(conf.newFeature,1)
    thisSample = 1;
    thisFrame  = 1;
    storeFrame = 2;
    for j=1:size(ldata.CLASS,1)
        if ldata.NEW_FEATURE(j,:) == conf.newFeature(i,:)
            if thisFrame > conf.trainStart && thisFrame <= conf.trainEnd
                tdata.input(thisSample, storeFrame               - 1 ) = ldata.DIFF_AVG(j,:);
                %tdata.input(thisSample, thisFrame               - 1 ) = ldata.AVG(j,:);
                %tdata.input(thisSample, thisFrame               - 1 ) = ldata.DIFF_STD(j,:);
                tdata.input(thisSample, storeFrame + itsFrames   - 2 ) = ldata.DIFF_STD(j,:);
                %tdata.input(thisSample, thisFrame + itsFrames   - 2 ) = ldata.STD(j,:);
                %tdata.input(thisSample, thisFrame               - 1 ) = ldata.DIFF_SPACE(j,:);
                tdata.input(thisSample, storeFrame + itsFrames*2 - 3 ) = ldata.DIFF_SPACE(j,:);
                    
                %tdata.input(thisSample, thisFrame               - 1 ) = ldata.MAXX(j,:);
                %tdata.input(thisSample, thisFrame + itsFrames   - 2 ) = ldata.MAXY(j,:);
                    
                %tdata.input(thisSample, thisFrame               - 1 ) = ldata.MAXY(j,:);
                storeFrame = storeFrame + 1;
            end
            
            if thisFrame == totalFrames
                tdata.target(thisSample,1) = ldata.CLASS(j,:) + 1;  
                thisSample = thisSample + 1;
                thisFrame  = 1;
                storeFrame = 2;
            else
                thisFrame = thisFrame + 1;
            end
        end
    end
end

tdata.TEST.target = tdata.target;

if(isfield(conf,'trainTwoLayer') && strcmp(conf.trainTwoLayer,'yes')) 
    sortMatrix = zeros(size(tdata.target,1) - conf.LNOsize,1);
    for i=1:size(tdata.target,1) - conf.LNOsize
        sortMatrix(i,:) = mod(i,2);
    end
end
    
for i=1:conf.LNOsize:size(tdata.input,1)
    
    fprintf('Train 1-%d and %d-%d\n',i-1,i+conf.LNOsize,size(tdata.input,1));
    fprintf('Test %d-%d\n',i,(i+conf.LNOsize - 1));
    
    TestIn  = tdata.input(i:(i+conf.LNOsize - 1),:); 
    TestOut = tdata.target(i:(i+conf.LNOsize - 1),:);
    
    if i > 1
        TrainInLower  = tdata.input(1:(i-1),:);
        TrainOutLower = tdata.target(1:(i-1),:);
    else
        TrainInLower  = [];
        TrainOutLower = [];
    end
    
    if i < size(ldata.CLASS,1) - conf.LNOsize
        TrainInUpper  = tdata.input(i+conf.LNOsize:size(tdata.input,1),:);
        TrainOutUpper = tdata.target(i+conf.LNOsize:size(tdata.input,1),:); 
    else
        TrainInUpper  = [];
        TrainOutUpper = [];
    end
    
    TrainIn  = [TrainInLower  ; TrainInUpper];
    TrainOut = [TrainOutLower ; TrainOutUpper];
    
    %--------------------------------------------------------------------------

    if(isfield(conf,'trainModelPCA') && strcmp(conf.trainModelPCA,'yes')) 
        [pn,meanp,stdp]   = nprestd(TrainIn');
        [OUTPUT,transMat] = prepca(pn,conf.trainModelPCAEigen); 
    
        pnewn             = trastd(TrainIn',meanp,stdp);
        OUTPUT            = trapca(pnewn,transMat);
        tdata.TRAIN.input = OUTPUT';
    
        pnewn             = trastd(TestIn',meanp,stdp);
        OUTPUT            = trapca(pnewn,transMat);
        tdata.TEST.input  = OUTPUT';
    end
    
    %--------------------------------------------------------------------------

    if(isfield(conf,'trainTwoLayer') && strcmp(conf.trainTwoLayer,'yes')) 
        [B,I] = sort(sortMatrix);
        TrainOutSort = TrainOut(I,:);
        TrainInSort  = TrainIn(I,:);
        
        TrainOutL1 = TrainOutSort(1:floor(size(TrainOutSort,1)/2),:);
        TrainOutL2 = TrainOutSort(floor(size(TrainOutSort,1)/2) + 1 :size(TrainOutSort,1),:);
        
        TrainInL1  = TrainInSort(1:floor(size(TrainInSort,1)/2),:);
        TrainInL2  = TrainInSort(floor(size(TrainInSort,1)/2) + 1 :size(TrainInSort,1),:);
        
        %--------------------------------------------------------------------------
        
        tdata.SVM.model1 = svmtrain(TrainOutL1, TrainInL1, conf.svmTrainOptions);
        
        [tdata.SVM.TrainPredict2a, tdata.SVM.TrainAccuracy, tdata.SVM.TrainValues] = ...
            svmpredict(TrainOutL2, TrainInL2, tdata.SVM.model1, conf.svmTestOptions);
        
        %Augment = tdata.SVM.TrainPredict2a - TrainOutL2;
        Augment = tdata.SVM.TrainPredict2a;
        
        tdata.SVM.model2 = svmtrain(TrainOutL2, [TrainInL2 Augment], conf.svmTrainOptions2);
        
        [tdata.SVM.TrainPredict2b, tdata.SVM.TrainAccuracy, tdata.SVM.TrainValues] = ...
            svmpredict(TrainOutL2, [TrainInL2 tdata.SVM.TrainPredict2a], tdata.SVM.model2, conf.svmTestOptions);
        
        %--------------------------------------------------------------------------
        
        [tdata.SVM.TestPredict1,   ...
         tdata.SVM.TestAccuracy,  ...
         tdata.SVM.TestValues] =  ...
        svmpredict(TestOut, TestIn, tdata.SVM.model1, conf.svmTestOptions); 
    
        %Augment = tdata.SVM.TestPredict1 - TestOut;
        Augment = tdata.SVM.TestPredict1;
    
        [tdata.SVM.TestPredict(i:(i+conf.LNOsize) - 1,:),   ...
         tdata.SVM.TestAccuracy,  ...
         tdata.SVM.TestValues] =  ...
        svmpredict(TestOut, [TestIn Augment], tdata.SVM.model2, conf.svmTestOptions);
        
    else
        %    fprintf('Using options for SVM \"%s\" \n',conf.svmTrainOptions);

        tdata.SVM.model = svmtrain(TrainOut, TrainIn, conf.svmTrainOptions);

        [tdata.SVM.TrainPredict, tdata.SVM.TrainAccuracy, tdata.SVM.TrainValues] = ...
            svmpredict(TrainOut, TrainIn, tdata.SVM.model, conf.svmTestOptions);
        %--------------------------------------------------------------------------

        [tdata.SVM.TestPredict(i:(i+conf.LNOsize) - 1,:),   ...
         tdata.SVM.TestAccuracy,  ...
         tdata.SVM.TestValues] =  ...
        svmpredict(TestOut, TestIn, tdata.SVM.model, conf.svmTestOptions);
    end
end   
    
    
    
    
