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

%conf.useLegacy    = 'no';                                                
%conf.condString  = 'UCIO_old';
%conf.condString  = 'UCIO_basic';          % Standard Surprise Model
%conf.condString  = 'UHIO_basic';         % Standard Surprise Model
%conf.condString  = 'UHIOLTWX_basic';     % Standard Surprise Model
%conf.condString  = 'NATHAN_UCIO_basic';  % Standard Surprise Model
%conf.condString  = 'JointGG_UCIO_basic'; % Standard Surprise Model
%conf.condString  = 'UCIO_opt';           % "optimized" parameters used
%conf.condString  = 'UHIO_opt';           % "optimized" parameters used
%conf.condString  = 'UHIOLTWX_opt';       % "optimized" parameters used
%conf.condString  = 'NATHAN_UCIO_opt';    % "optimized" parameters used
%conf.condString  = 'JointGG_UCIO_opt';   % "optimized" parameters used
conf.useLegacy    = 'yes';
%conf.condString  = 'UCIO_legacy';        % Requires an old copy of ezvision
%conf.condString  = 'UHIO_legacy';        % Requires an old copy of ezvision
%conf.condString  = 'UHIOLTX_legacy';     % Requires an old copy of ezvision
conf.condString  = 'UHIOGKSE_legacy';    % Requires an old copy of ezvision

conf.type        = 'lin_ind';                               % Type of classifier
conf.typeLabel   = 'Linear-Ind';                            % output label
conf.skipFeature = 'no';                                   % If yes we will not test against the training features
conf.doTestSet   = 'yes';                                   % Interleave the training and testing sets
conf.testPerl    = 'no';                                    % If yes, we will test the perl parser
conf.graphRegression       = 'yes';                         % Should we graph the class and correlation results?
conf.graphBasic            = 'yes';                         % Graph basic results?
conf.regressionBasic       = 'yes';                         % Run basic regression?
conf.regressionNorm        = 'yes';                         % Run normalized regression?
conf.regressionBiased      = 'yes';                         % Run Biased regression
conf.regressionConstrained = 'yes';                         % Run constrained regression 
conf.regressionUseBound    = 'yes';                         % Keep training bounded samples out of the test?
conf.graphClasses          = 'yes';                         % graph the format of each class
conf.runHardSet            = 'no';                         % should we include the extra hard set data?
conf.runNormalSet          = 'yes';                          % should we run the normal set? 
conf.hardBound   = 1;                                       % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
conf.easyBound   = 8;                                       % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
conf.baseDir     = '/lab/mundhenk/linear-classifier/';      % Base directory for all this
conf.binary      = '/lab/mundhenk/saliency/bin/ezvision';   % Where is the surprise binary 
conf.procRSVP    = [conf.baseDir 'script/process_rsvp.pl']; % script called by process-em.sh and beoqueue
%conf.imageDir    = '/lab/raid/images/RSVP/fullSequence/';   % where are the images to be processed
%conf.imageDir    = '/lab/tmpib/30/fullSequence/';
conf.imageDir    = '/lab/mundhenk/AnimTransDistr/Hard.Link.Sequences/EEG.2.11.2008/';
conf.inputStr     = '\t\"--in=$d/stim#.png --input-frames=0-MAX\\@$fdur \". \n';

%conf.imagePrefix = 'stim??_???';                            % beoqueue image name prefix
conf.imagePrefix = 'stim*'; 
conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
                    'n05 n05 n05 n05 n07 n07 n07 n07 ' ...
                    'n08 n08 n08 n08  ' ...
                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo'];
%conf.nodes       = ['n01 n01 n01 n01 n02 n02 n02 n02 ' ...
%                    'n03 n03 n03 n03 n04 n04 n04 n04 ' ...
%                    'n05 n05 n05 n05 n07 n07 n07 n07 ' ...
%                    'n08 n08 n08 n08  ']; 
%conf.nodes       = ['ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ' ...
%                    'ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo'];
conf.beoqueue    = [conf.baseDir 'script/beoqueue.pl'];                % home of beoqueue script
conf.runLocalOnly = 'no';                                  % if true, we will only run locally via basic shell and not rsh
conf.procEM      = [conf.baseDir 'script/process-em.sh'];   % where to create the short process-em script

conf.surprise.duration    = '50ms';                         % duration of frame in surprise
conf.surprise.useStandard = 'yes';                          % use a standard surprise model
conf.surprise.logCommand  = 'yes';                           % set to yes to record each ezvision command call in process_RSVP
conf.surpriseOnly         = 'yes';                          % if defined and set to yes, we will not run the matlab analysis
conf.extraPerlArgs        = '-sh';                           % Extra args to pass to the perl post processor if any
conf.RSVP_extra_args      = '';                             % Extra args sent to surprise, used mostly during optimizations
conf.meta_options         = '';

% Set up channels
conf = linear_model_set_up_channels(conf);

% Run the classifier
ldata = linear_model(conf);

