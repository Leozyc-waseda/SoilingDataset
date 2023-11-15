function [conf,tdata] = train_model_create_sets(ldata,tdata,conf)

itsFrames = 11;

% do some initial work on class sorting
feature_label = conf.feature_label;
feature_num   = conf.feature_num;

tdata.NEW_CLASS = ldata.NEW_CLASS;

%for i=1:size(ldata.CLASS,1) 
%    % For testing purposes
%    if     ldata.CLASS(i,1) < conf.hardBound
%        tdata.NEW_CLASS(i,1) = 1;
%    elseif ldata.CLASS(i,1) < conf.easyBound
%        tdata.NEW_CLASS(i,1) = 2;
%    else
%        tdata.NEW_CLASS(i,1) = 3;
%    end
%end

tdata.NEW_FEATURE   = zeros(size(ldata.CLASS,1),1);
tdata.feature_label = feature_label;

for i=1:size(ldata.CLASS,1)
    for j=1:feature_num
        if strcmp(ldata.FEATURE(i,1),feature_label{j})
            tdata.NEW_FEATURE(i,1) = j;
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

[tdata.CLASS,tdata.SET,tdata.SET_MEMBER,tdata.SWITCH] = train_model_split_sets(ldata.SAMPLE,ldata.TFRAME,ldata.CLASS,2);

% resize data structures
%tdata.CLASS      = ldata.CLASS + 1;
%tdata.SET_MEMBER = zeros(size(tdata.CLASS,1),1) - 1;
%tdata.SET        = zeros(max(max(ldata.SAMPLE)),max(max(ldata.TFRAME))) - 1;
%tswitch          = zeros(max(max(tdata.CLASS)),1) - 1;

% assign each sequence to one of two classes, but balance the set based on
% target class to keep them balanced

%ltswitch = 0;

%for i=1:size(tdata.CLASS,1)
    % New image sequence set?
%    if tdata.SET(ldata.SAMPLE(i,1),ldata.TFRAME(i,1)) == -1
        % switch the switch itself
%        if tswitch(tdata.CLASS(i,1)) == -1
%            tswitch(tdata.CLASS(i,1)) = ltswitch;
%            if ltswitch == 0
%                ltswitch = 1;
%            else
%                ltswitch = 0;
%            end
%        end
        % place in one of two data sets keeping class target balanced
%    	tdata.SET(ldata.SAMPLE(i,1),ldata.TFRAME(i,1)) = tswitch(tdata.CLASS(i,1));
        % switch to other class to keep balance
%        if tswitch(tdata.CLASS(i,1)) == 1
%            tswitch(tdata.CLASS(i,1)) = 0;
%        else
%            tswitch(tdata.CLASS(i,1)) = 1;
%        end
%    end
%    tdata.SET_MEMBER(i,:) = tdata.SET(ldata.SAMPLE(i,1),ldata.TFRAME(i,1));
%end

% take all the data and create two literal data sets
trainSize = 1;
testSize  = 1;

tdata.TRAIN.CLASS            = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.NEW_CLASS        = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.NEW_FEATURE      = zeros(size(tdata.CLASS,1)/2,1);
%tdata.TRAIN.FEATURE         = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.TFRAME           = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.SAMPLE           = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.AVG              = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.STD              = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.DIFF_AVG         = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.DIFF_STD         = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.DIFF_SPACE       = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.DIFF_TARG_AVG    = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.DIFF_TARG_STD    = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.DIFF_TARG_SPACE  = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.MAXX             = zeros(size(tdata.CLASS,1)/2,1);
tdata.TRAIN.MAXY             = zeros(size(tdata.CLASS,1)/2,1);

tdata.TEST.CLASS             = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.NEW_CLASS         = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.NEW_FEATURE       = zeros(size(tdata.CLASS,1)/2,1);
%tdata.TEST.FEATURE          = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.TFRAME            = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.SAMPLE            = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.AVG               = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.STD               = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.DIFF_AVG          = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.DIFF_STD          = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.DIFF_SPACE        = zeros(size(tdata.CLASS,1)/2,1); 
tdata.TEST.DIFF_TARG_AVG     = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.DIFF_TARG_STD     = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.DIFF_TARG_SPACE   = zeros(size(tdata.CLASS,1)/2,1);      
tdata.TEST.MAXX              = zeros(size(tdata.CLASS,1)/2,1);
tdata.TEST.MAXY              = zeros(size(tdata.CLASS,1)/2,1);

% Put data into the two sets
for i=1:size(tdata.CLASS,1)
    % training set
    if tdata.SET_MEMBER(i,:) == 0
        tdata.TRAIN.CLASS(trainSize,:)            = tdata.CLASS(i,:);
        tdata.TRAIN.NEW_CLASS(trainSize,:)        = tdata.NEW_CLASS(i,:);
        tdata.TRAIN.NEW_FEATURE(trainSize,:)      = tdata.NEW_FEATURE(i,:);
        tdata.TRAIN.FEATURE(trainSize,:)          = ldata.FEATURE(i,:);
        tdata.TRAIN.TFRAME(trainSize,:)           = ldata.TFRAME(i,:);
        tdata.TRAIN.NEWFRAME(trainSize,:)         = ldata.TFRAME(i,:) - 5;
        tdata.TRAIN.SAMPLE(trainSize,:)           = ldata.SAMPLE(i,:);
        tdata.TRAIN.AVG(trainSize,:)              = ldata.AVG(i,:);
        tdata.TRAIN.STD(trainSize,:)              = ldata.STD(i,:);
        tdata.TRAIN.DIFF_AVG(trainSize,:)         = ldata.DIFF_AVG(i,:);
        tdata.TRAIN.DIFF_STD(trainSize,:)         = ldata.DIFF_STD(i,:);
        tdata.TRAIN.DIFF_SPACE(trainSize,:)       = ldata.DIFF_SPACE(i,:);
        tdata.TRAIN.DIFF_TARG_AVG(trainSize,:)    = ldata.DIFF_TARG_AVG(i,:);
        tdata.TRAIN.DIFF_TARG_STD(trainSize,:)    = ldata.DIFF_TARG_STD(i,:);
        tdata.TRAIN.DIFF_TARG_SPACE(trainSize,:)  = ldata.DIFF_TARG_SPACE(i,:);
        tdata.TRAIN.MAXX(trainSize,:)             = ldata.MAXX(i,:);
        tdata.TRAIN.MAXY(trainSize,:)             = ldata.MAXY(i,:);
        trainSize = trainSize + 1;
    else
        tdata.TEST.CLASS(testSize,:)              = tdata.CLASS(i,:);
        tdata.TEST.NEW_CLASS(testSize,:)          = tdata.NEW_CLASS(i,:);
        tdata.TEST.NEW_FEATURE(testSize,:)        = tdata.NEW_FEATURE(i,:);
        tdata.TEST.FEATURE(testSize,:)            = ldata.FEATURE(i,:);
        tdata.TEST.TFRAME(testSize,:)             = ldata.TFRAME(i,:);
        tdata.TEST.NEWFRAME(testSize,:)           = ldata.TFRAME(i,:) - 5;
        tdata.TEST.SAMPLE(testSize,:)             = ldata.SAMPLE(i,:);
        tdata.TEST.AVG(testSize,:)                = ldata.AVG(i,:);
        tdata.TEST.STD(testSize,:)                = ldata.STD(i,:);
        tdata.TEST.DIFF_AVG(testSize,:)           = ldata.DIFF_AVG(i,:);
        tdata.TEST.DIFF_STD(testSize,:)           = ldata.DIFF_STD(i,:);
        tdata.TEST.DIFF_SPACE(testSize,:)         = ldata.DIFF_SPACE(i,:);  
        tdata.TEST.DIFF_TARG_AVG(testSize,:)      = ldata.DIFF_TARG_AVG(i,:);
        tdata.TEST.DIFF_TARG_STD(testSize,:)      = ldata.DIFF_TARG_STD(i,:);
        tdata.TEST.DIFF_TARG_SPACE(testSize,:)    = ldata.DIFF_TARG_SPACE(i,:);
        tdata.TEST.MAXX(testSize,:)               = ldata.MAXX(i,:);
        tdata.TEST.MAXY(testSize,:)               = ldata.MAXY(i,:);
        testSize = testSize + 1;
    end
end

% For each set, align into a standard matlab training matrix




