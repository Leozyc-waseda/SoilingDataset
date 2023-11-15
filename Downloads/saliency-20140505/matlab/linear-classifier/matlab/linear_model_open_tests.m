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

% Use this file for creating some open tests


%function f = linear_model()
ldata         = struct('Description','Holds the data used in this script');
conf          = struct('Description','Holds configuration values');
conf.surprise = struct('Description','Holds values that control surprise execution');


conf.gatherStats = 'no';                        % Run the model in surprise from scratch
                                                % if you have never run
                                                % this before you will want
                                                % to select yes to run the
                                                % surprise binary

% Condition string. You can make your own. Make sure you also edit the 
% condition string in "process-RSVP.pl" so they match (yes its hacky)                                        
                                                
conf.condString  = 'UCIO_basic';
%conf.condString  = 'UCIO_old';
%conf.condString  = 'UHIO_basic';
%conf.condString  = 'NATHAN_UCIO_basic';
%conf.condString  = 'JointGG_UCIO_basic';
%conf.condString  = 'UCIO_legacy';

conf.type        = 'lin_ind';                               % Type of classifier
conf.type_label  = 'Linear-Ind';                            % output label
conf.skipFeature = 'no';                                    % If yes we will not test against the training features
conf.hardBound   = 2;                                       % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
conf.easyBound   = 7;                                       % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
conf.baseDir     = '/lab/mundhenk/linear-classifier/';      % Base directory for all this
conf.binary      = '/lab/mundhenk/saliency/bin/ezvision';   % Where is the surprise binary 
conf.procRSVP    = [conf.baseDir 'script/process_rsvp.pl']; % script called by process-em.sh and beoqueue
conf.imageDir    = '/lab/raid/images/RSVP/fullSequence/';   % where are the images to be processed
conf.imagePrefix = 'stim??_???';                            % beoqueue image name prefix
conf.nodes       = 'n01 n01 n01 n01 n02 n02 n02 n02 n02 n03 n03 n03 n03 n04 n04 n04 n04 n05 n05 n05 n05 n07 n07 n07 n07 n08 n08 n08 n08 ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo ibeo';
conf.beoqueue    = '$HOME/rsvp/beoqueue.pl';
conf.procEM      = [conf.baseDir 'script/process-em.sh'];

conf.surprise.duration    = '50ms';                         % duration of frame in surprise
conf.surprise.useStandard = 'yes';                          % use a standard surprise model
  
tic;
fprintf('<CREATE> Making process-em file\t- %s\n',conf.procEM); 
conf = linear_model_build_process_EM(conf);
t = toc; fprintf('\t\t<time %f>\n',t);

tic;     
fprintf('<CREATE> Making process_RSVP.pl file\t- %s\n',conf.procRSVP); 
conf = linear_model_build_process_RSVP(conf);
t = toc; fprintf('\t\t<time %f>\n',t);