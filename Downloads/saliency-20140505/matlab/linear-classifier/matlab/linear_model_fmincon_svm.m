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

function fval = linear_classify_fmincon_svm(X)

persistent FLAST;
global COUNT;
global ERROR_STATE;


conf           = struct('Description','Holds configuration values');
conf.startTime = clock;
conf.surprise  = struct('Description','Holds values that control surprise execution');

conf.gatherStats = 'yes';                        % Run the model in surprise from scratch
                                                % if you have never run
                                                % this before you will want
                                                % to select yes to run the
                                                % surprise binary

% Condition string. You can make your own. Make sure you also edit the 
% condition string in "linear_model_build_process_RSVP.m" so they match (yes its hacky)                                        

conf.useLegacy    = 'no';                                                

%conf.condString  = 'UHIO_train';                % Standard Surprise Model
%conf.condString  = 'UHIOLTWXE_train';           % Standard Surprise Model
conf.condString  = 'UHIOLTXE_train';           % Standard Surprise Model

%TRAIN CHAN
% fmincon
%conf.trainStr      = ['\t\"--surprise --vc-type=UH:' num2str(X(1,1)) 'I:' num2str(X(1,2)) 'O:' num2str(X(1,3)) ' \". \n'];
%conf.trainStr      = ['\t\"--surprise --vc-type=UH:' num2str(X(1,1)) 'I:' num2str(X(1,2)) 'O:' num2str(X(1,3)) ...
%                                                'L:' num2str(X(1,4)) 'T:' num2str(X(1,5)) 'W:' num2str(X(1,6)) ...
%                                                'X:' num2str(X(1,7)) 'E:'
%                                                num2str(X(1,8)) ' \".
%                                                \n'];

conf.trainStr      = ['\t\"--surprise --vc-type=UH:' num2str(X(1,1)) 'I:' num2str(X(1,2)) 'O:' num2str(X(1,3)) ...
                                                'L:' num2str(X(1,4)) 'T:' num2str(X(1,5)) 'X:' num2str(X(1,6)) ...
                                                'E:' num2str(X(1,7)) ' \". \n'];
% fminsearch
%conf.trainStr      = ['\t\"--surprise --vc-type=UH:' num2str(X(1,1)) 'I:' num2str(X(2,1)) 'O:' num2str(X(3,1)) ' \". \n'];

%TRAIN FAC
% Error = 0.2179999999999999993338661852249060757458209991455078125000000000
%C = [ 0.8773092344262305442015303924563340842723846435546875000000000000 ...
%      0.3127783654225734788489887705509318038821220397949218750000000000 ...
%      1.5660499968021082128899479357642121613025665283203125000000000000 ];

%conf.trainChanStr  = ['\t\"--surprise --vc-type=UH:' num2str(C(1,1)) 'I:' num2str(C(1,2)) 'O:' num2str(C(1,3))];
%conf.trainFacStr1  = [' --surprise-slfac='    num2str(X(1,1)) ' --surprise-ssfac='  num2str(X(1,2))];
%conf.trainFacStr2  = [' --surprise-neighsig=' num2str(X(1,3)) ' --surprise-locsig=' num2str(X(1,4)) ' \". \n'];

%conf.trainStr = [conf.trainChanStr conf.trainFacStr2];

fprintf('\n\n******************************************\n');
fprintf('RUNNING %s\n',conf.trainStr);
fprintf('******************************************\n');

conf.type        = 'lin_ind';                               % Type of classifier
conf.typeLabel   = 'Linear-Ind';                            % output label
conf.skipFeature = 'yes';                                    % If yes we will not test against the training features
conf.doTestSet   = 'yes';                                    % Interleave the training and testing sets
conf.testPerl    = 'no';                                    % If yes, we will test the perl parser
conf.getFrequencyData = 'no';                              % run and gather fourier spectrum on conspic maps
conf.graphRegression       = 'yes';                         % Should we graph the class and correlation results?
conf.graphBasic            = 'no';                          % Graph basic results?
conf.regressionBasic       = 'yes';                         % Run basic regression?
conf.regressionNorm        = 'yes';                         % Run normalized regression?
conf.regressionBiased      = 'yes';                         % Run Biased regression
conf.regressionConstrained = 'yes';                         % Run constrained regression 
conf.regressionUseBound    = 'yes';                          % Keep training bounded samples out of the test?
conf.graphClasses          = 'no';                         % graph the format of each class
conf.graphDiffStats        = 'no';                         % graph the differential stats rather than basic stats
conf.graphCombinedStats    = 'no';                          % Should we graph combined stats?
conf.graphConspicStats     = 'no';                          % Should we bar graph conspicuity stats?
conf.graphCombinedSumStats = 'no';                         % Should we graph the combined stats summed for all subject responses
conf.runHardSet            = 'no';                         % should we include the extra hard set data?
conf.runNormalSet          = 'yes';                          % should we run the normal set?            
conf.hardBound   = 1;                                       % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
conf.easyBound   = 8;                                       % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
conf.midBound    = 5;
conf.trainDataUse          = 'diff';                      % base, diff or target
conf.trainModelComplex     = 'yes';                         % Use the SVM training method
conf.trainModelEasyHard    = 'no';                         % use the three class system rather than the 9 class system
conf.trainModelPCA         = 'no';                         % Use PCA on the data?
conf.trainModelPCAEigen    = 0.001;                          % PCA Eigen Value cutoff
conf.trainLNO              = 'no';                        % Use Leave N Out training and testing
conf.trainTwoLayer         = 'no';                          % use two iterative SVM's
conf.trainStart            = 1;                           % Starting bound for samples in frame
conf.trainEnd              = 11;                          % Starting bound for samples in frame
conf.trainRound            = 'yes';                        % Check to clamp and round output values
conf.LNOsize               = 10;                           % Leave N out interval size
conf.baseDir     = '/lab/mundhenk/linear-classifier/';      % Base directory for all this

conf.binary      = '/lab/mundhenk/saliency/bin/ezvision';   % Where is the surprise binary 
conf.procRSVP    = [conf.baseDir 'script/process_rsvp.pl']; % script called by process-em.sh and beoqueue
%conf.imageDir    = '/lab/raid/images/RSVP/fullSequence/';   % where are the images to be processed
conf.imageDir    = '/lab/tmpib/u/fullSequence/';
conf.hardDir     = '/lab/tmpib/u/hardSequence/';
conf.imagePrefix = 'stim??_???';                            % beoqueue image name prefix
%conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
%                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
%                    'n05 n05 n05 n05 n06 n06 n06 n06 ' ...
%                    'n07 n07 n07 n07 n08 n08 n08 n08 ' ...
%                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
%                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo'];
conf.nodes       = [ 'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
                     'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo'];
conf.runLocalOnly = 'yes'; 

if(strcmp(conf.getFrequencyData,'yes'))
	conf.nodes       = [ 'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
                         'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo'];  
    conf.runLocalOnly = 'no'; 
end
    
%conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
%                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
%                    'n05 n05 n05 n05 n07 n07 n07 n07 ' ...
%                    'n08 n08 n08 n08  '];
conf.beoqueue    = [conf.baseDir 'script/beoqueue.pl'];    % home of beoqueue script
%conf.beoqueue    = '$HOME/rsvp/beoqueue.pl';                % home of beoqueue script
                                 % if true, we will only run locally via basic shell and not rsh
conf.procEM      = [conf.baseDir 'script/process-em.sh'];   % where to create the short process-em script

conf.surprise.duration    = '50ms';                         % duration of frame in surprise
conf.surprise.useStandard = 'yes';                          % use a standard surprise model
conf.surprise.logCommand  = 'yes';                           % set to yes to record each ezvision command call in process_RSVP
conf.RSVP_extra_args      = '';                             % Extra args sent to surprise, used mostly during optimizations
conf.meta_options         = '';

% if training on specific features list them here
conf.trainFeatures   = 1;
conf.featureTrain{1} = 'final';

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

if strcmp(conf.condString,'UHIO_opt')         || strcmp(conf.condString,'UCIO_opt')      || ...
   strcmp(conf.condString,'UHIOLTWX_opt')     || strcmp(conf.condString,'UHIOLTXE_opt')  || ...
   strcmp(conf.condString,'NATHAN_UCIO_opt')  || strcmp(conf.condString,'UHIOLTWXE_opt') || ...
   strcmp(conf.condString,'Outlier_UHIO_opt')

    % Default is:
    % (temporal) slfac    = 1.0 
    % (space)    ssfac    = 0.1 
    %            neighsig = 0.5
    %            locsig   = 3.0 
    %conf.RSVP_extra_args       = '\t\"--surprise-qlen=1 \". \n';
    conf.RSVP_extra_args       = '\t\"--surprise-slfac=0.1 --surprise-ssfac=1 --surprise-neighsig=0.5 --surprise-locsig=3 \". \n';
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
end

conf.meta_options.meta_log     = [conf.baseDir 'log/min_trainlog.txt'];
conf.meta_options.prec_in_log  = [conf.baseDir 'log/min_trainparam.txt'];
conf.meta_options.prec_out_log = [conf.baseDir 'log/min_trainfval.txt'];

time = clock;
command = ['echo RUNNING ' num2str(COUNT) ': ' conf.RSVP_extra_args ' :     ',date,' ',...
        num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
        ' >> ' conf.meta_options.meta_log];
unix(command);

flog = fopen(conf.meta_options.prec_in_log,'a');
fprintf(flog,'%d\t',COUNT);
for i=1:size(X,2)
    fprintf(flog,'%4.64f\t',X(1,i));
end
fprintf(flog,'\n');
fclose(flog);

% Set up channels
conf = linear_model_set_up_channels(conf);

% Run the classifier
[ldata,tdata,ftdata] = linear_model(conf);

fval = ftdata{10,2}.dat.RSVP.finalError;

fprintf('\n\n******************************************\n');
fprintf('TRAIN ERROR %f\n',fval);
fprintf('******************************************\n');

time = clock;
command = ['echo ERROR ' num2str(COUNT) ': ' num2str(fval)  ... 
           ' :     ' date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
           ' >> ' conf.meta_options.meta_log];

unix(command);  
flog = fopen(conf.meta_options.prec_out_log,'a');
fprintf(flog,'%d\t%4.64f\n',COUNT,fval);
fclose(flog);

FLAST = fval;
COUNT = COUNT + 1;






