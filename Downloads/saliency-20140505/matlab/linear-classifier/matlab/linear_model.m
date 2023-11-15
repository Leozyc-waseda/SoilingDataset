% //////////////////////////////////////////////////////////////////// %
%           Surprise Linear Model - Copyright (C) 2004-2007            %
% by the University of Southern California (USC) and the iLab at USC.  %
% See http://iLab.usc.edu for information about this project.          %
% //////////////////////////////////////////////////////////////////// %
% This file is part of the iLab Neuromorphic Vision Toolkit            %
%                                                                      %
% The Surprise Linear Model is free software; you can                  %
% redistribute it and/or modify it under the terms of the GNU General  %
% Public License as published by the Free Software Foundation; either  %
% version 2 of the License, or (at your option) any later version.     %
%                                                                      %
% The Surprise Linear Model is distributed in the hope                 %
% that it will be useful, but WITHOUT ANY WARRANTY; without even the   %
% implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      %
% PURPOSE.  See the GNU General Public License for more details.       %
%                                                                      %
% You should have received a copy of the GNU General Public License    %
% along with the iBaysian Surprise Matlab Toolkit; if not, write       %
% to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   %
% Boston, MA 02111-1307 USA.                                           %
% //////////////////////////////////////////////////////////////////// %
%
% Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
% $Revision: 55 $ 
% $Id$
% $HeadURL: https://surprise-mltk.svn.sourceforge.net/svnroot/surprise-mltk/source/surprise_toolkit/example_graph.m $
function [ldata,tdata,ftdata] = linear_model(conf)

ldata          = struct('Description','Holds the data used in this script');
tdata          = struct('Description','Holds the data two training and testing sets');

% We cannot run hard/base sets at the same time as mask sets
if(isfield(conf,'runMaskSet') && strcmp(conf.runMaskSet,'yes')) 
    if(isfield(conf,'runNormalSet') && strcmp(conf.runNormalSet,'yes'))
        error('Cannot run Mask and Regular set at the same time');
    elseif(isfield(conf,'runHardSet') && strcmp(conf.runHardSet,'yes')) 
        error('Cannot run Mask and Hard set at the same time');
    end
end

% Call to place files into /lab/tmpib/30 if needed
%[unp_status,unp_result] = unix('/lab/mundhenk/linear-classifier/script/unpack.sh');
if strcmp(conf.gatherStats,'yes')
    gacheck = 0;
    dprint('Gathering stats from ezvision');
    if(isfield(conf,'runNormalSet') && strcmp(conf.runNormalSet,'yes'))
        conf = Gather_Stats('normal',conf);
        gacheck = 1;
    end
    % run the hard set data hard_pre hard_post hard_w if requested
    if(isfield(conf,'runHardSet') && strcmp(conf.runHardSet,'yes')) 
        conf = Gather_Stats('hard',conf);
        gacheck = 1;
    end
    % run the mask set data 
    if(isfield(conf,'runMaskSet') && strcmp(conf.runMaskSet,'yes')) 
        conf = Gather_Stats('mask',conf);
        gacheck = gacheck + 1;
    end
    % run the mask set data
    if(isfield(conf,'runTransMaskSet') && strcmp(conf.runTransMaskSet,'yes')) 
        conf = Gather_Stats('trans_mask',conf);
        gacheck = gacheck + 1;
    end
    % run the hard mask set data
    if(isfield(conf,'runHardMaskSet') && strcmp(conf.runHardMaskSet,'yes')) 
        conf = Gather_Stats('hard_mask',conf);
        gacheck = gacheck + 1;
    end
    % run the not-hard mask set data
    if(isfield(conf,'runNotHardMaskSet') && strcmp(conf.runNotHardMaskSet,'yes')) 
        conf = Gather_Stats('not-hard_mask',conf);
        gacheck = gacheck + 1;
    end
    % run the new trans set 
    if(isfield(conf,'runNewTransSet') && strcmp(conf.runNewTransSet,'yes')) 
        conf = Gather_Stats('trans',conf);
        gacheck = gacheck + 1;
    end
    if gacheck == 0
        error('Type of stats to gather from ezvision not given');
    elseif gacheck > 1
        error('Cannot run many sets at the same time');
    end
end

% Post process raw data using perl
if strcmp(conf.gatherStats,'yes') || strcmp(conf.testPerl,'yes')
    conf = linear_model_post_process_perl(conf);
end
    
% Do we just want to run surprise only to gather data, but not run the
% matlab analysis?
if ~isfield(conf,'surpriseOnly') || strcmp(conf.surpriseOnly,'no')
  
    % Read in the raw data after perl processing
    [ldata,conf] = linear_model_post_read_data(ldata,conf);
    
    %%%%%%%%%%%%%%%%%%%
    % Call linear_model
    %%%%%%%%%%%%%%%%%%%
    dprint('Running Linear Classify');

    % (1a) Compute Normalized Stats
    dprint('Compute normalized diff stats');
    tprint('start');
    ldata.DIFF_AVG          = linear_model_diff_stats(ldata.AVG,  ldata.NEWFRAME);
    tprint('stop');
    tprint('start');
    ldata.DIFF_STD          = linear_model_diff_stats(ldata.STD,  ldata.NEWFRAME);
    tprint('stop');
    tprint('start');
    ldata.DIFF_SPACE        = linear_model_diff_space(ldata.MAXX, ldata.MAXY, ldata.NEWFRAME);
    tprint('stop');
    
    % (1b) Compute target offset Stats
    dprint('Compute offset diff target stats');
    tprint('start');
    ldata.DIFF_TARG_AVG     = linear_model_difftarg_stats(ldata.AVG, ldata.NEWFRAME);
    tprint('stop');
    tprint('start');
    ldata.DIFF_TARG_STD     = linear_model_difftarg_stats(ldata.STD, ldata.NEWFRAME);
    tprint('stop');
    tprint('start');
    ldata.DIFF_TARG_SPACE   = linear_model_difftarg_space(ldata.MAXX, ldata.MAXY, ldata.NEWFRAME);
    tprint('stop');
    
    % (2) Train The Model
    
    tprint('start');
    % If we are using the certanty metric, use it as the bound
    if(isfield(conf,'useCert') && strcmp(conf.useCert,'yes'))
        dprint('Train Model Cert');
        conf.hardBound  = conf.certHard; 
        conf.easyBound  = conf.certEasy;
    else
        dprint('Train Model');
    end
    
    % get Easy / Hard classification
    ldata.NEW_CLASS = linear_model_get_new_class(ldata.CLASS,conf);
    
    % get various model statistics (this is not really training)
    ldata.MODEL = linear_model_train(ldata.DIFF_AVG,ldata.DIFF_STD,ldata.DIFF_SPACE,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    tprint('stop');   
    
    % If requested, graph the class data of the different classes using the
    % basic bar method with sig value. This isn't nessesary for the full
    % script to run.
    if (isfield(conf,'graphClasses') && strcmp(conf.graphClasses,'yes'))
        ldata = combined_stats(ldata,conf);
    end
    
    % at this point we either use the linear classifier or the complex
    % trainer
    if(isfield(conf,'trainLNO') && strcmp(conf.trainLNO,'yes')) 
        dprint('TRAIN : Using training option Train Leave One Out');
        [tdata,ldata,conf] = Leave_One_Out_SVM(tdata,ldata,conf);
        ftdata             = struct('Description','Holds the data two training and testing sets - per feature');
    elseif(isfield(conf,'trainModelComplex') && strcmp(conf.trainModelComplex,'yes')) % complex train
        dprint('TRAIN : Using training option Complex SVM One Out');
        [tdata,ldata,ftdata,conf] = Complex_SVM(tdata,ldata,conf);
    elseif(isfield(conf,'trainModelLinear') && strcmp(conf.trainModelLinear,'yes')) % linear classifier
        dprint('TRAIN : Using training option Train Linear Model Out');
        [tdata,ldata,conf] = Linear_Template(tdata,ldata,conf);
        ftdata             = struct('Description','Holds the data two training and testing sets - per feature');
    elseif(isfield(conf,'trainAttentionGate') && strcmp(conf.trainAttentionGate,'yes')) % Attention Gate classifier
        dprint('TRAIN : Using training on attention gate');
        [tdata,ldata,conf] = Attention_Gate(tdata,ldata,conf);
        ftdata             = struct('Description','Holds the data two training and testing sets - per feature');
    else
        dprint('TRAIN : Using NO training');
        ftdata             = struct('Description','Holds the data two training and testing sets - per feature');
    end

    conf.endTime = clock;
else
    dprint('NOTICE : Skiping matlab analysis');
end


%--------------------------------------------------------------------------
function [tdata,ldata,conf] = Leave_One_Out_SVM(tdata,ldata,conf)

tprint('start');
tdata = train_model_leave_n_out(ldata,conf);
tprint('stop');
        
dprintf('RSVP test');
tprint('start');
tdata = train_model_rsvp_test(tdata,9,conf);
tprint('stop');

%--------------------------------------------------------------------------
function [tdata,ldata,conf] = Attention_Gate(tdata,ldata,conf)

% create two set indexes
if (isfield(conf,'trainModelEasyHard') && strcmp(conf.trainModelEasyHard,'no')) 
    [tdata.CLASS,tdata.SET,tdata.SET_MEMBER,tdata.SWITCH] = train_model_split_sets(ldata.M_SAMPLE,ldata.M_TFRAME,ldata.M_CLASS,2);
    c_num = 9;
else
    [tdata.CLASS,tdata.SET,tdata.SET_MEMBER,tdata.SWITCH] = train_model_split_sets(ldata.M_SAMPLE,ldata.M_TFRAME,ldata.M_NEW_CLASS,2);
    c_num = 3;
end

tdata.TRAIN.target            = zeros(size(tdata.CLASS,1)/(2*11),1);
tdata.TEST.target             = zeros(size(tdata.CLASS,1)/(2*11),1);
tdata.TRAIN.input             = zeros(size(tdata.CLASS,1)/(2*11),2);
tdata.TEST.input              = zeros(size(tdata.CLASS,1)/(2*11),2);

% Put data into the two sets
trainSize = 1; testSize = 1;
for i=1:size(tdata.CLASS,1)
    % training set
    if ldata.M_NEWFRAME(i,:) == 6 % just get the central target frame
        if tdata.SET_MEMBER(i,:) == 0
            tdata.TRAIN.target(trainSize,:)            = tdata.CLASS(i,:);
            tdata.TRAIN.input(trainSize,:)             = [ldata.LAM_AVG(i,1) ldata.LAM_STD(i,1)]; 
            trainSize = trainSize + 1;
        else
            tdata.TEST.target(testSize,:)              = tdata.CLASS(i,:);
            tdata.TEST.input(testSize,:)               = [ldata.LAM_AVG(i,1) ldata.LAM_STD(i,1)]; 
            testSize = testSize + 1;
        end
    end
end
% Run RSVP data
tdata = train_model_svm(tdata,conf);
% Test results
%tdata = train_model_rsvp_test(tdata,c_num,conf);

%--------------------------------------------------------------------------
function [tdata,ldata,ftdata,conf] = Complex_SVM(tdata,ldata,conf)

% (3) Create Train-Test sets
dprint('Create Train-Test sets');
tprint('start');  
[conf,tdata] = train_model_create_sets(ldata,tdata,conf);
tdata        = train_model_finish_sets(ldata,tdata,conf);
tprint('stop');
        
ftdata{conf.feature_num + 1,1} = 'Output vals';
ftdata{conf.feature_num + 2,1} = 'Rounded vals';
        
for i=1:conf.feature_num
	conf.newFeature = i;
	fprintf('\n**********************\n');
	dprint(['USING Feature ' conf.feature_label{i}]);
	ftdata{i,1} = conf.feature_label{i};
	ftdata{i,2}.dat = struct('Description','Holds the data two training and testing sets - per feature');
    
    % (4) Create Train-Test sets
    dprint('Finish Train-Test sets');
    tprint('start');  
    ftdata{i,2}.dat = train_model_finish_sets(ldata,tdata,conf);
    tprint('stop');
        
    % (5) SVM Train and Test
    dprint('Train Test SVM');
    tprint('start');
    ftdata{i,2}.dat = train_model_svm(ftdata{i,2}.dat,conf);
    tprint('stop');
        
    % (6) Run RSVP test on model
    if (isfield(conf,'trainModelEasyHard') && strcmp(conf.trainModelEasyHard,'no')) 
    	dprint('RSVP test');
        tprint('start');
        ftdata{i,2}.dat = train_model_rsvp_test(ftdata{i,2}.dat,9,conf);
        tprint('stop');
        csize = 9;
    else
    	dprint('training Easy-Hard only'); 
        dprint('RSVP test');
        tprint('start');
        ftdata{i,2}.dat = train_model_rsvp_test(ftdata{i,2}.dat,3,conf);
        tprint('stop');
        csize = 3;
    end
    
	ftdata{i,3} = ftdata{i,2}.dat.RSVP.finalError;
    ftdata{conf.feature_num + 1,2} = [ftdata{conf.feature_num + 1,2} ftdata{i,2}.dat.RSVP.preRoundPredict];
    ftdata{conf.feature_num + 2,2} = [ftdata{conf.feature_num + 2,2} ftdata{i,2}.dat.SVM.TestPredict];
 end
        
dprint('Combine Test SVM');
tprint('start');
tdata = train_model_combine_factors(ftdata,tdata,csize,conf);
tprint('stop');
        
dprint('Vote test');
tdata.VOTE = tdata;
tdata.VOTE.SVM.TestPredict = tdata.COMBINE.Vote;
tdata.VOTE = train_model_rsvp_test(tdata.VOTE,csize,conf);
        
dprint('Certanty test');
tdata.CERT = tdata;
tdata.CERT.SVM.TestPredict = tdata.COMBINE.Cert;
tdata.CERT = train_model_rsvp_test(tdata.CERT,csize,conf); 
        
dprint('Average test');
tdata.AVG = tdata;
tdata.AVG.SVM.TestPredict = tdata.COMBINE.Avg;
tdata.AVG = train_model_rsvp_test(tdata.AVG,csize,conf);


%--------------------------------------------------------------------------
function [tdata,ldata,conf] = Linear_Template(tdata,ldata,conf)

% (3) Classify Data Points
fprintf('>>>Classify\n');
tprint('start');
ldata.NEW_CLASS         = linear_model_classify(ldata.MODEL,ldata.DIFF_AVG,ldata.DIFF_STD,ldata.DIFF_SPACE,ldata.NEWFRAME,ldata.TFRAME,ldata.SAMPLE,ldata.FEATURE,ldata.CLASS,conf);
tprint('stop');

% (4) Final claasify for testing
fprintf('>>>Final Class\n');
tprint('start');
ldata.FINAL_CLASS       = linear_model_final_classify(ldata.NEW_CLASS,conf);
tprint('stop');

% (5) Analyze Results - T
fprintf('>>>Analyze T\n');
tprint('start');
ldata.LINEAR_TEST_STATS = linear_model_analyze(ldata.FINAL_CLASS);
tprint('stop');

% (6) Analyze Results - Regression
fprintf('>>>Analyze R\n');
tprint('start');
ldata.LINEAR_TEST_STATS = linear_model_test_regression(ldata.FINAL_CLASS,ldata.LINEAR_TEST_STATS,conf);
tprint('stop');

if strcmp(conf.graphBasic,'yes') 
    % (7) Graph Results
    fprintf('>>>Graph\n');
    tprint('start');
    ldata.R                 = linear_model_graph_error(ldata.LINEAR_TEST_STATS,conf);
    tprint('stop');
end

fprintf('DONE\n');


%--------------------------------------------------------------------------
function conf = Gather_Stats(condition,conf)

%set up 'process_RSVP.pl' file for running    
tprint('start');
conf = linear_model_build_process_RSVP(conf,condition);
tprint('stop');
        
%set up 'process-em.sh' file for running  
tprint('start');
conf = linear_model_build_process_EM(conf,condition);
tprint('stop'); 
        
% Call surprise on image set
command = ['sh ' conf.procEM];
dprint(['Process ' condition ' SET EM - ' command]);
tprint('start');
[em_status,em_result] = unix(command,'-echo');
tprint('stop');


