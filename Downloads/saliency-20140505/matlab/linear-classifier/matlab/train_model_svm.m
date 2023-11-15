function tdata = train_model_svm(tdata,conf)

%--------------------------------------------------------------------------

% SVM constants - Three Class Ideal
%Degree     = 5;
%C          = 1;
%Gamma      = 1; 
%Coeff      = 1;
%Weight     = [ 1 1 1 ];

%--------------------------------------------------------------------------
tdata.SVM = struct('description','Holder for SVM data');
% By using this format, the default values of Gamma, Coefficient,
% u, Epsilon, CacheSize are used. 
% That is, Gamma=1, Coefficient=1, u=0.5, Epsilon=0.001, and CacheSize=45MB
%[tdata.SVM.AlphaY, tdata.SVM.SVs, tdata.SVM.Bias, tdata.SVM.Parameters, tdata.SVM.nSV, tdata.SVM.nLabel] = ...
%    PolySVC(tdata.TRAIN.input', tdata.TRAIN.target', conf.svm.Degree, conf.svm.C, conf.svm.Gamma, conf.svm.Coeff);
%[tdata.SVM.AlphaY, tdata.SVM.SVs, tdata.SVM.Bias, tdata.SVM.Parameters, tdata.SVM.nSV, tdata.SVM.nLabel] = ...
%    PolySVC(tdata.TRAIN.input', tdata.TRAIN.target');

if(isfield(conf,'trainModelPCA') && strcmp(conf.trainModelPCA,'yes')) 
    dprint('Using PCA');
    [pn,meanp,stdp]   = nprestd(tdata.TRAIN.input');
    [OUTPUT,transMat] = prepca(pn,conf.trainModelPCAEigen); 
    
    pnewn             = trastd(tdata.TRAIN.input',meanp,stdp);
    OUTPUT            = trapca(pnewn,transMat);
    tdata.TRAIN.input = OUTPUT';
    
    pnewn             = trastd(tdata.TEST.input',meanp,stdp);
    OUTPUT            = trapca(pnewn,transMat);
    tdata.TEST.input  = OUTPUT';
end


fprintf('Using options for SVM \"%s\" \n',conf.svmTrainOptions);

tdata.SVM.model = svmtrain(tdata.TRAIN.target, tdata.TRAIN.input, conf.svmTrainOptions);

%[AlphaY, SVs, Bias, Parameters, nSV, nLabel] = RbfSVC(ptrans, transpose(trainOut), Gamma, C);
%save SVMClassifier AlphaY SVs Bias Parameters nSV nLabel;

%load SVMClassifier;

%--------------------------------------------------------------------------

%[tdata.SVM.TClassRate, tdata.SVM.TDecisionValue, tdata.SVM.TNs, tdata.SVM.TConfMatrix, tdata.SVM.TPreLabels] = ...
%    SVMTest(tdata.TRAIN.input', tdata.TRAIN.target', tdata.SVM.AlphaY, tdata.SVM.SVs, tdata.SVM.Bias, tdata.SVM.Parameters, tdata.SVM.nSV, tdata.SVM.nLabel);

%[tdata.SVM.TrainPredict, tdata.SVM.T] = ...
%    SVMClass(tdata.TRAIN.input', tdata.SVM.AlphaY, tdata.SVM.SVs, tdata.SVM.Bias, tdata.SVM.Parameters, tdata.SVM.nSV, tdata.SVM.nLabel);

[tdata.SVM.TrainPredict, tdata.SVM.TrainAccuracy, tdata.SVM.TrainValues] = ...
    svmpredict(tdata.TRAIN.target, tdata.TRAIN.input, tdata.SVM.model, conf.svmTestOptions);
%--------------------------------------------------------------------------

%[tdata.SVM.YClassRate, tdata.SVM.YDecisionValue, tdata.SVM.YNs, tdata.SVM.YConfMatrix, tdata.SVM.YPreLabels] = ...
%    SVMTest(tdata.TEST.input', tdata.TEST.target', tdata.SVM.AlphaY, tdata.SVM.SVs, tdata.SVM.Bias, tdata.SVM.Parameters, tdata.SVM.nSV, tdata.SVM.nLabel);

%[tdata.SVM.TestPredict, tdata.SVM.Y] = ...
%    SVMClass(tdata.TEST.input', tdata.SVM.AlphaY, tdata.SVM.SVs, tdata.SVM.Bias, tdata.SVM.Parameters, tdata.SVM.nSV, tdata.SVM.nLabel);

[tdata.SVM.TestPredict, tdata.SVM.TestAccuracy, tdata.SVM.TestValues] = ...
    svmpredict(tdata.TEST.target, tdata.TEST.input, tdata.SVM.model, conf.svmTestOptions);
