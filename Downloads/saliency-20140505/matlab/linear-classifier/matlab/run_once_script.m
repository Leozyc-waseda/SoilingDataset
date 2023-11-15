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

% used to hold some debug data
% We use a global since we want to be able to pass around some data ad hoc when
% we debug so we don't get stuck trying to pass data we really don't need around
% in more standardized data structures. 
global DEBUG;

conf           = struct('Description','Holds configuration values');
conf.startTime = clock;
conf.surprise  = struct('Description','Holds values that control surprise execution');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %IMPORTANT: This parameter can save you time! 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.gatherStats = 'yes';                       % Run the model in surprise from scratch
                                                % if you have never run
                                                % this before you will want
                                                % to select yes to run the
                                                % surprise binary
                                                % OTHERWISE...
                                                % we can reuse the
                                                % statistics from a
                                                % previous run
                                                
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Save EVERYTHING to an m file?
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.saveFinalData     = 'no';        % Save all the final data in an m file
conf.saveFinalDataName = '/lab/mundhenk/linear-classifier/nothardmask.mat'; % what to call the final file        

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Choose ezvision configuration
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
% Condition string. You can make your own. Make sure you also edit the 
% condition string in "linear_model_build_process_RSVP.m" so they match (yes its hacky)                                        

conf.useLegacy        = 'no';   
conf.useSaliencyOnly  = 'no';                % run Saliency only and not surprise
%conf.condString  = 'UCIO_old';
%conf.condString  = 'UCIO_basic';              % Standard Surprise Model
%conf.condString  = 'UHO_basic';              % Standard Surprise Model
%conf.condString  = 'UHOX_basic';              % Standard Surprise Model
%conf.condString  = 'UHOT_basic';              % Standard Surprise Model
%conf.condString  = 'UHOL_basic';              % Standard Surprise Model
%conf.condString  = 'UHOE_basic';              % Standard Surprise Model
%conf.condString  = 'UHOW_basic';              % Standard Surprise Model
%conf.condString  = 'UHIO_basic';              % Standard Surprise Model
%conf.condString  = 'UHIOLTWX_basic';          % Standard Surprise Model
%conf.condString  = 'UHIOLTWXE_basic';         % Standard Surprise Model
%conf.condString  = 'UHIOLTXE_basic';           % Standard Surprise Model
%conf.condString  = 'UHOLTXE_basic';           % Standard Surprise Model (H = H2SV2 Color)
%conf.condString  = 'UQOLTXE_basic';           % Standard Surprise Model (Q = CIELab Color)
%conf.condString  = 'UCIOLTXE_basic';           % Standard Surprise Model (CI = RG-BY and Intensity Color)
conf.condString  = 'UIOLTXE_basic';           % Standard Surprise Model >>>No color<<<
%conf.condString  = 'UHIOE_basic';             % Standard Surprise Model
%conf.condString  = 'UHIOL_basic';             % Standard Surprise Model
%conf.condString  = 'UHIOT_basic';             % Standard Surprise Model
%conf.condString  = 'UHIOX_basic';             % Standard Surprise Model
%conf.condString  = 'ChiSq_UHIO_basic';         % Standard Surprise Model
%conf.condString  = 'ChiSq_UHIOLTXE_basic';     % Standard Surprise Model
%conf.condString  = 'ChiSq_UHOLTXE_basic';      % Standard Surprise Model
%conf.condString  = 'UHIO_max';                % Max Surprise Model
%conf.condString  = 'UHIOLTXE_max';            % Max Surprise Model
%conf.condString  = 'UHOLTXE_max';             % Max Surprise Model
%conf.condString  = 'UQOLTXE_max';             % Max Surprise Model
%conf.condString  = 'UCIOLTXE_max';             % Max Surprise Model
%conf.condString  = 'NATHAN_UCIO_basic';       % Standard Surprise Model
%conf.condString  = 'PoissonConst_UCIO_basic';      % Standard Surprise Model
%conf.condString  = 'PoissonConst_UHIO_basic';      % Standard Surprise Model
%conf.condString  = 'PoissonConst_UHIOLTXE_basic';  % Standard Surprise Model
%conf.condString  = 'PoissonConst_UHOLTXE_basic';  % Standard Surprise Model
%conf.condString  = 'PoissonConst_UQOLTXE_basic';  % Standard Surprise Model
%conf.condString  = 'PoissonConst_UCIOLTXE_basic';  % Standard Surprise Model
%conf.condString  = 'PoissonFloat_UHOLTXE_basic';  % Standard Surprise Model
%conf.condString  = 'JointGG_UCIO_basic';      % Standard Surprise Model
%conf.condString  = 'JointGG_UHIO_basic';      % Standard Surprise Model
%conf.condString  = 'JointGG_UHIOLTXE_basic';  % Standard Surprise Model
%conf.condString  = 'JointGG_UHOLTXE_basic';  % Standard Surprise Model
%conf.condString  = 'Gaussian_UHIO_basic';     % Gauss Surprise Model
%conf.condString  = 'Gaussian_UHOLTXE_basic';     % Gauss Surprise Model
%conf.condString  = 'Outlier_UHIO_basic';      % Outlier Surprise Model
%conf.condString  = 'Outlier_UHOLTXE_basic';    % Outlier Surprise Model
%conf.condString  = 'UCIO_opt';                % "optimized" parameters used
%conf.condString  = 'UHIO_opt';                % "optimized" parameters used
%conf.condString  = 'UHIOLTWX_opt';            % "optimized" parameters used
%conf.condString  = 'UHIOLTWXE_opt';           % "optimized" parameters used
%conf.condString  = 'UHIOLTXE_opt';            % "optimized" parameters used
%conf.condString  = 'UHOLTXE_opt';            % "optimized" parameters used
%conf.condString  = 'UQOLTXE_opt';            % "optimized" parameters used
%conf.condString  = 'NATHAN_UCIO_opt';         % "optimized" parameters used
%conf.condString  = 'JointGG_UCIO_opt';        % "optimized" parameters used
%conf.condString  = 'JointGG_UHIO_opt';        % "optimized" parameters used
%conf.condString  = 'JointGG_UHIOLTXE_opt';    % "optimized" parameters used
%conf.condString  = 'Outlier_UHIO_opt';        % Outlier Surprise Model

%conf.useLegacy   = 'yes';
%conf.condString  = 'UCIO_legacy';             % Requires an old copy of ezvision
%conf.condString  = 'UHIO_legacy';             % Requires an old copy of ezvision
%conf.condString  = 'UHIOLTX_legacy';          % Requires an old copy of ezvision
%conf.condString  = 'UHIOGKSE_legacy';         % Requires an old copy of ezvision

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set basic matlab params/configs
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.type                  = 'lin_ind';                     % Type of classifier
conf.typeLabel             = 'Linear-Ind';                  % output label
conf.skipFeature           = 'yes';                         % If yes we will not test against the training features
conf.doTestSet             = 'yes';                         % Interleave the training and testing sets
conf.testPerl              = 'no';                          % If yes, we will test the perl parser
conf.getFrequencyData      = 'no';                          % run and gather fourier spectrum on conspic maps

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Graphing Params
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.graphRegression       = 'yes';                         % Should we graph the class and correlation results?
conf.graphBasic            = 'yes';                         % Graph basic results?   
conf.graphClasses          = 'yes';                         % graph the format of each class
conf.graphDiffStats        = 'no';                          % graph the differential stats rather than basic stats
conf.graphTargStats        = 'no';                          % Graph target centric diff stats
conf.graphBasicStats       = 'yes';                         % Graph channel stats, no diff
conf.graphCombinedStats    = 'no';                          % Should we graph combined stats?
conf.graphHistStats        = 'yes';                         % Should we graph the basic histogram over avg, std surprise in the diff?
conf.graphSpaceStats       = 'no';                          % Should we graph the space stats?
conf.graphConspicStats     = 'no';                          % Should we bar graph conspicuity stats?
conf.graphCombinedSumStats = 'no';                          % Should we graph the combined stats summed for all subject responses
conf.graphABClasses        = 'no';                          % Graph classes for AB statistics in AB data
conf.graphABOffsets        = 'no';                         % Graph final data for AB at each offset frame
conf.graphAGStats          = 'no';                          % Graph attention gate stats?

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Special Statistical Tests
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.useBonfCorrect        = 'yes';                         % Use Bonferoni correction
conf.usePooledStd          = 'no';                          % Use a pooled std error
conf.usePooledByFrame      = 'yes';                         % if pooling std err, should we only do it within a frame
conf.printT                = 'yes';                         % print t-test results   
conf.runPairedTests        = 'no';                          % Should we run paired T tests?
conf.runMWTests            = 'no';                          % Run differential tests between neighbors?
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Training Params
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.regressionBasic       = 'yes';                         % Run basic regression?
conf.regressionNorm        = 'yes';                         % Run normalized regression?
conf.regressionBiased      = 'yes';                         % Run Biased regression
conf.regressionConstrained = 'yes';                         % Run constrained regression 
conf.regressionUseBound    = 'yes';                         % Keep training bounded samples out of the test?    
conf.trainDataUse          = 'diff';                        % base, diff or target
conf.trainAttentionGate    = 'no';                          % Should we do training based on the attention gata data?
conf.trainModelLinear      = 'no';                          % Use the original linear classifier?
conf.trainModelComplex     = 'no';                          % Use the SVM training method
conf.trainModelEasyHard    = 'yes';                         % use the three class system rather than the 9 class system
conf.trainModelPCA         = 'no';                          % Use PCA on the data?
conf.trainModelPCAEigen    = 0.001;                         % PCA Eigen Value cutoff
conf.trainLNO              = 'no';                          % Use Leave N Out training and testing
conf.trainTwoLayer         = 'no';                          % use two iterative SVM's
conf.trainStart            = 1;                             % Starting bound for samples in frame
conf.trainEnd              = 11;                            % Starting bound for samples in frame
conf.trainRound            = 'yes';                         % Check to clamp and round output values
conf.LNOsize               = 10;                            % Leave N out interval size

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run a special set?
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.runHardSet            = 'no';                         % should we include the extra hard set data?
conf.runNormalSet          = 'no';                         % should we run the normal set?        
conf.runMaskSet            = 'no';                        % Animals with masks
conf.runTransMaskSet       = 'yes';                         % Transportation with masks
conf.runHardMaskSet        = 'no';                         % Special Hard Animals with masks
conf.runNotHardMaskSet     = 'no';                         % Special Easy Animals with masks
conf.runNewTransSet        = 'no';                         % should we run the new trans set? 
conf.runABSet              = 'no';                         % This is the AB set and will call a somewhat different set of m files

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set cluster data processing params / paths
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
conf.baseDir         = '/lab/mundhenk/linear-classifier/';      % Base directory for all this
conf.binary          = '/lab/mundhenk/saliency/bin/ezvision';   % Where is the surprise binary 
conf.procRSVP        = [conf.baseDir 'script/process_rsvp.pl']; % script called by process-em.sh and beoqueue
conf.beoqueue        = [conf.baseDir 'script/beoqueue.pl'];     % home of beoqueue script
%conf.beoqueue       = '$HOME/rsvp/beoqueue.pl';                % home of beoqueue script
conf.procEM          = [conf.baseDir 'script/process-em.sh'];   % where to create the short process-em script

% Image file prefix
if(strcmp(conf.runABSet,'yes'))
    conf.imagePrefix = 'stim_??_??_???';  
else
    conf.imagePrefix = 'stim??_???';        % beoqueue image name prefix (AB has a different prefix)
end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Location of special directories for different data sets
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%conf.imageDir       = '/lab/raid/images/RSVP/fullSequence/';   % where are the images to be processed
conf.imageDir        = '/lab/tmpib/u/nathan/fullSequence/';
conf.hardDir         = '/lab/tmpib/u/nathan/hardSequence/';
conf.maskDir         = '/lab/tmpib/u/nathan/maskSequence/';
conf.transMaskDir    = '/lab/tmpib/u/nathan/newSequences/TransMask/';
conf.hardMaskDir     = '/lab/tmpib/u/nathan/maskSequenceHard/';
%conf.hardMaskDir     = '/lab/tmpib/u/nathan/maskSequenceHard-Select/';
conf.notHardMaskDir  = '/lab/tmpib/u/nathan/maskSequenceNotHard-Select/';
conf.transDir        = '/lab/tmpib/u/nathan/newSequences/TransOnly/';
conf.ABDir           = '/lab/tmpib/u/nathan/newSequences/Partial-Repete-AB-Var-T1/';
% conf.xxxBound moved to linear_model_post_read_data.m

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Select how to run on the cluster
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conf.runLocalOnly = 'no'; % Run only on the local machine?
                          % if true, we will only run locally via basic 
                          % shell and not rsh
 
%conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
%                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
%                    'n05 n05 n05 n05 n06 n06 n06 n06 ' ...
%                    'n07 n07 n07 n07 n08 n08 n08 n08 ' ...
%                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
%                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
%                   ];

%conf.nodes       = [ 'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
%                     'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
%                   ]; 

conf.nodes       = ['icore icore icore icore icore icore icore icore ' ...
                    'icore icore icore icore icore icore icore icore ' ...
                    'icore icore icore icore icore icore icore icore ' ...
                    'icore icore icore icore icore icore icore icore ' ...
                   ];

if(strcmp(conf.getFrequencyData,'yes'))
	conf.nodes       = [ 'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
                         'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo'];  
    conf.runLocalOnly = 'no'; 
end
    
%conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
%                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
%                    'n05 n05 n05 n05 n07 n07 n07 n07 ' ...
%                    'n08 n08 n08 n08  '];


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Extra surprise commands sent to command line
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                               

conf.surprise.duration    = '50ms';                         % duration of frame in surprise
conf.surprise.useStandard = 'yes';                          % use a standard surprise model
conf.surprise.logCommand  = 'yes';                          % set to yes to record each ezvision command call in process_RSVP
conf.RSVP_extra_args      = '';                             % Extra args sent to surprise, used mostly during optimizations
conf.meta_options         = '';

% if training on specific features list them here
conf.trainFeatures   = 1;
conf.featureTrain{1} = 'final';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set SVM training params if used
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% support vector parameters
%conf.svm.Degree     = 5;
%conf.svm.C          = 1;
%conf.svm.Gamma      = 0.25; 
%conf.svm.Coeff      = 0.5;

%options:
%-s svm_type : set type of SVM (default 0)%
%	0 -- C-SVC
%	1 -- nu-SVC
%	2 -- one-class SVM
%	3 -- epsilon-SVR
%	4 -- nu-SVR
%-t kernel_type : set type of kernel function (default 2)
%	0 -- linear: u'*v
%	1 -- polynomial: (gamma*u'*v + coef0)^degree
%	2 -- radial basis function: exp(-gamma*|u-v|^2)
%	3 -- sigmoid: tanh(gamma*u'*v + coef0)
%-d degree : set degree in kernel function (default 3)
%-g gamma : set gamma in kernel function (default 1/k)
%-r coef0 : set coef0 in kernel function (default 0)
%-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
%-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
%-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
%-m cachesize : set cache memory size in MB (default 100)
%-e epsilon : set tolerance of termination criterion (default 0.001)
%-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
%-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
%-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)

%conf.svmTrainOptions = '-t 0';
%conf.svmTrainOptions = '-t 1 -d 2 -s 4 -b 1 -m 1000';
conf.svmTrainOptions = '-t 1 -d 2 -m 1000';
%conf.svmTrainOptions = '-t 1 -d 3';
%conf.svmTrainOptions = '-t 1 -d 4';
%conf.svmTrainOptions = '-t 2';
%conf.svmTrainOptions = '-t 3';
conf.svmTestOptions  = '';

conf.svmTrainOptions2 = '-t 0';

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set extra parameters for optimization/training runs
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
if strcmp(conf.condString,'UHIO_opt')         || strcmp(conf.condString,'UCIO_opt')      || ...
   strcmp(conf.condString,'UHIOLTWX_opt')     || strcmp(conf.condString,'UHIOLTXE_opt')  || ...
   strcmp(conf.condString,'NATHAN_UCIO_opt')  || strcmp(conf.condString,'UHIOLTWXE_opt') || ...
   strcmp(conf.condString,'Outlier_UHIO_opt') || strcmp(conf.condString,'UHOLTXE_opt')   || ...
   strcmp(conf.condString,'UQOLTXE_opt')

    % Default is:
    % (temporal) slfac    = 1.0 
    % (space)    ssfac    = 0.1 
    %            neighsig = 0.5
    %            locsig   = 3.0 
    %conf.RSVP_extra_args       = '\t\"--surprise-qlen=1 \". \n';
    %conf.RSVP_extra_args       = '\t\"--surprise-slfac=0.1 --surprise-ssfac=1.0 --surprise-neighsig=0.5 --surprise-locsig=3 \". \n';
    conf.RSVP_extra_args       = '\t\"--surprise-slfac=0.25 --surprise-ssfac=1.0 --surprise-neighsig=0.5 --surprise-locsig=3 \". \n';
    %conf.RSVP_extra_args       = ['\t\"' ...
    %	'--surprise-slfac=0.0596697360278561628188498389135929755866527557373046875000000000 '...	
    %    '--surprise-ssfac=0.9883349830289824833329248576774261891841888427734375000000000000 '...	
    %    '--surprise-neighsig=0.4062493465396932457167622487759217619895935058593750000000000000	'...
    %    '--surprise-locsig=2.9608824388866148424881430401001125574111938476562500000000000000 '...
    %    '\". \n'];
    %conf.RSVP_extra_args       = ['\t\"' ...
    %	'--surprise-slfac=-0.1711924565549051424628856921117403544485569000244140625000000000 '...	
    %    '--surprise-ssfac=0.1851035868840598119788865005830302834510803222656250000000000000 '...	
    %    '--surprise-neighsig=1.7521361387560441258415266929659992456436157226562500000000000000	'...
    %    '--surprise-locsig=1.6248059042198723656014180960482917726039886474609375000000000000 '...
    %    '\". \n'];
    %conf.RSVP_extra_args       = ['\t\"' ...
    %	'--surprise-slfac=-0.1702652083259778570401010711066192016005516052246093750000000000 '...	
    %    '--surprise-ssfac=0.1892484850178392241648595017977640964090824127197265625000000000 '...	
    %    '--surprise-neighsig=1.7618858712427374335618424083804711699485778808593750000000000000 '...
    %    '--surprise-locsig=1.711299614177810735071716408128850162029266357421875000000000000 '...
    %    '\". \n'];
    % Abs value set
    %conf.RSVP_extra_args       = ['\t\"' ...
    %	'--surprise-slfac=-0.1063055478951265842013640394725371152162551879882812500000000000 '...	
    %    '--surprise-ssfac=0.5723246142099625011212538083782419562339782714843750000000000000 '...	
    %    '--surprise-neighsig=0.5061156308856045171751247835345566272735595703125000000000000000 '...
    %    '--surprise-locsig=1.2008054804327541464914475000114180147647857666015625000000000000 '...
    %    '\". \n'];
elseif strcmp(conf.condString,'JointGG_UCIO_opt') || strcmp(conf.condString,'JointGG_UHIO_opt') || strcmp(conf.condString,'JointGG_UHIOLTXE_opt')
    conf.RSVP_extra_args       = '\t\"--surprise-kl-bias=Static --surprise-slfac=1.0 --surprise-ssfac=0.1 --surprise-neighsig=0.5 --surprise-locsig=3 \". \n';      
elseif strcmp(conf.condString,'UHIO_max') || strcmp(conf.condString,'UHIOLTXE_max') || strcmp(conf.condString,'UHOLTXE_max')
    conf.RSVP_extra_args       = '\t\"--surprise-take-st-max --surprise-slfac=1.0 --surprise-ssfac=1.0 --surprise-neighsig=0.5 --surprise-locsig=3 \". \n';  
end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Run this puppy!!!!
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Set up channels
conf = linear_model_set_up_channels(conf);

% Run the classifier
if(strcmp(conf.runABSet,'yes'))
    [ldata,tdata,ftdata] = linear_model_AB(conf);
else
    [ldata,tdata,ftdata] = linear_model(conf);
end

if strcmp(conf.saveFinalData,'yes')
    save(conf.saveFinalDataName);
end

