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

function [fval,gval] = linear_classify_fmincon(X)

persistent FLAST;
global COUNT;
global ERROR_STATE;

conf           = struct('Description','Holds configuration values');
conf.startTime = clock;
conf.surprise  = struct('Description','Holds values that control surprise execution');

conf.gatherStats = 'yes';                       % Run the model in surprise from scratch
                                                % if you have never run
                                                % this before you will want
                                                % to select yes to run the
                                                % surprise binary

% Condition string. You can make your own. Make sure you also edit the 
% condition string in "linear_model_build_process_RSVP.m" so they match (yes its hacky)                                        
                                                
%conf.condString  = 'UCIO_basic';
%conf.condString  = 'UCIO_old';
conf.condString  = 'UHIO_basic';
%conf.condString  = 'NATHAN_UCIO_basic';
%conf.condString  = 'JointGG_UCIO_basic';
%conf.condString  = 'UCIO_legacy';

conf.simFMINCON  = 'no';                                  % If yes then create sham data to test fmincon
conf.type        = 'lin_ind';                               % Type of classifier
conf.typeLabel   = 'Linear-Ind';                            % output label
conf.skipFeature = 'no';                                   % If yes we will not test against the training features
conf.doTestSet   = 'yes';                                   % Interleave the training and testing sets
conf.testPerl    = 'no';                                    % If yes, we will test the perl parser
conf.graphRegression       = 'no';                         % Should we graph the class and correlation results?
conf.graphBasic            = 'no';                         % Graph basic results?
conf.regressionBasic       = 'no';                         % Run basic regression?
conf.regressionNorm        = 'yes';                         % Run normalized regression?
conf.regressionBiased      = 'no';                         % Run Biased regression
conf.regressionConstrained = 'no';                         % Run constrained regression 
conf.regressionUseBound    = 'yes';                         % Keep training bounded samples out of the test?
conf.hardBound   = 1;                                       % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
conf.easyBound   = 8;                                       % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
conf.baseDir     = '/lab/mundhenk/linear-classifier/';      % Base directory for all this
conf.binary      = '/lab/mundhenk/saliency/bin/ezvision';   % Where is the surprise binary 
conf.procRSVP    = [conf.baseDir 'script/process_rsvp.pl']; % script called by process-em.sh and beoqueue
%conf.imageDir    = '/lab/raid/images/RSVP/fullSequence/';   % where are the images to be processed
conf.imageDir    = '/lab/tmpib/u/fullSequence/';
conf.imagePrefix = 'stim??_???';                            % beoqueue image name prefix
conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
                    'n05 n05 n05 n05 n07 n07 n07 n07 ' ...
                    'n08 n08 n08 n08  ' ...
                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
                    'ibeo ibeo ibeo ibeo'];
%conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
%                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
%                    'n05 n05 n05 n05 n07 n07 n07 n07 ' ...
%                    'n08 n08 n08 n08  '];
conf.beoqueue    = '$HOME/rsvp/beoqueue.pl';                % home of beoqueue script
conf.procEM      = [conf.baseDir 'script/process-em.sh'];   % where to create the short process-em script

conf.surprise.duration    = '50ms';                         % duration of frame in surprise
conf.surprise.useStandard = 'yes';                          % use a standard surprise model
conf.surprise.logCommand  = 'yes';                           % set to yes to record each ezvision command call in process_RSVP

% Extra args sent to surprise, used mostly during optimizations
[conf.RSVP_extra_args,conf.meta_options] = linear_model_get_extra_option_string(X);

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
ldata = linear_model(conf);

if ERROR_STATE.regression.isError == 0
    % Re can compute a sham error to test fmincon
    if strcmp(conf.simFMINCON,'yes')
        rmse = ldata.LINEAR_TEST_STATS.regnorm.rmse + rand(1)*0.1;
        b    = abs(0.5 - (7*ldata.LINEAR_TEST_STATS.regnorm.b(2,1))) + rand(1)*0.1;
    else
        rmse = ldata.LINEAR_TEST_STATS.regnorm.rmse;
        b    = abs(0.5 - (7*ldata.LINEAR_TEST_STATS.regnorm.b(2,1)));
    end

    % error is quadratic to encorage the laging variable
    %fval = rmse^2 + 2*rmse*b + b^2;
    fval = rmse^2 + b^2; 
    %fval = rmse;
    % Quadratic gradiant
    %gtmp = [2*rmse + b ; 2*b + rmse];

    %gtmp = 2*rmse + b + 2*b + rmse;
    gtmp = 2*rmse + 2*b;

    %gtmp = fval - FLAST;

    time = clock;
    command = ['echo ERROR ' num2str(COUNT) ': ' num2str(fval) ' RMSE ' num2str(rmse) ' b ' num2str(b) ... 
               ' GRAD ' num2str(gtmp(1,1)) ' :     ' date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
               ' >> ' conf.meta_options.meta_log];

else
    fval = 1;
    gtmp = 0;
    time = clock;
    command = ['echo ERROR : CATCH REGRESSION ERROR :' num2str(COUNT) ': ' num2str(fval) ... 
               ' GRAD ' num2str(gtmp(1,1)) ' :     ' date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
               ' >> ' conf.meta_options.meta_log];
end

unix(command);  
flog = fopen(conf.meta_options.prec_out_log,'a');
fprintf(flog,'%d\t%4.64f\n',COUNT,fval);
fclose(flog);

FLAST = fval;
COUNT = COUNT + 1;

gval = gtmp;




