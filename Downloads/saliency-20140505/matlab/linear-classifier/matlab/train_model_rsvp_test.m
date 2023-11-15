function tdata = train_model_rsvp_test(tdata,csize,conf)

tdata.RSVP = struct('description','Holds RSVP predictoin data. This is the method used in JOSA');

% first create the confusion matrix
% What the model predicts - tdata.SVM.TestPredict
% What it should get      - tdata.TEST.target

tdata.RSVP.confusionMatrix = zeros(csize,csize);
tdata.RSVP.expHitsMatrix   = zeros(csize,csize);
%tdata.TEST.target
%tdata.SVM.TestPredict

if isfield(conf,'trainRound') && strcmp(conf.trainRound,'yes')
    tdata.RSVP.preRoundPredict = tdata.SVM.TestPredict;
    tdata.SVM.TestPredict = round(tdata.SVM.TestPredict);
    for i=1:size(tdata.SVM.TestPredict,1)
        if tdata.SVM.TestPredict(i,:) > csize
           tdata.SVM.TestPredict(i,:) = csize;
        elseif tdata.SVM.TestPredict(i,:) < 1
           tdata.SVM.TestPredict(i,:) = 1;
        end
    end
end
    
    
for i = 1:size(tdata.SVM.TestPredict,1)
    tdata.RSVP.confusionMatrix(tdata.TEST.target(i,:),tdata.SVM.TestPredict(i,:)) = ...
        tdata.RSVP.confusionMatrix(tdata.TEST.target(i,:),tdata.SVM.TestPredict(i,:)) + 1;
end

% create row target sums
tdata.RSVP.targetSum = sum(tdata.RSVP.confusionMatrix,2);

% create target prediction
for i = 1:size(tdata.RSVP.targetSum,1);
    tdata.RSVP.targetProb(i,:) = ((i-1)/(csize - 1));
end

tdata.RSVP.expTargetHits = tdata.RSVP.targetSum .* tdata.RSVP.targetProb;

% create expected hits matrix
for i = 1:size(tdata.RSVP.confusionMatrix,1)
    tdata.RSVP.expHitsMatrix(i,:) = tdata.RSVP.confusionMatrix(i,:) .* tdata.RSVP.targetProb'; 
end

tdata.RSVP.expOutputHits = sum(tdata.RSVP.expHitsMatrix,2);

% find prediction error
tdata.RSVP.predictError = abs(tdata.RSVP.expOutputHits - tdata.RSVP.expTargetHits);
tdata.RSVP.sumError     = sum(sum(tdata.RSVP.predictError));
tdata.RSVP.finalError   = tdata.RSVP.sumError / size(tdata.SVM.TestPredict,1);

fprintf('RSVP Error is %f\n',tdata.RSVP.finalError);


