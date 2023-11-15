%**************************************************************
% The program is distributed freely under the GNU Public License.  
% See gpl.txt for details on usage/transference and editing the code.  
%
% Tested on Matlab 7.3+ (R2006b and above)
% Written by David Berg, John Shen and Laurent Itti	
% Currently maintained by David Berg and John Shen
% E-mail: dberg@usc.edu, shenjohn@usc.edu
% Last release: Nov 2009
%**************************************************************


%**************************************************************
% The eyeMarkup eye movement analysis tool
%**************************************************************
% This software is a flexible and easily extendable tool for analysis
% of eye movement data for a variety of scientific and commercial
% applications. Analysis is performed by defining a set of 'analysis
% modules' in an 'analysis pipeline'. The user chooses a collection of
% modules (either from the 'modules' sub-directory or from a user
% defined path) which are performed in order on the data. Results are
% appended to an internal data structure and passed to the next
% module, which may use the output of a previous module. In this way,
% the user can use a single program with different modules to perform
% a variety of standard and custom analysis. Modules can perform
% processing as simple as, for example, saccade detection using
% instantaneous velocity, or complex analysis such as rejection of
% found saccades based on the eye movement metrics.
% **************************************************************


% WHAT'S NEW?
%*****************************************
% A summary of the new usage:
% Input files (.eye in the lab) can be taken as before, as a
% three-column text file with (x, y, pupil diameter) data for each
% sample.
% (When running markEye in batch, parameters such as the sampling
% rate can be changed for each file by adding a header such as 
% # sf 200
% to the top of the .eye file.
% 
% Output is automatically given as a .e-ceyeS file with four header
% lines:
% period = #Hz 
% ppd = ##.#
% trash = #
% cols = x y pd status *targetx *targety ...
% The cols line gives the fields of the next rows.  Asterisks are
% given to those columns which are almost always zero except during 
% the beginning of a saccade.  This is done to combine the columns you want.
% The rest of the output is a space-delimited table of the stats.
%
% To change the output behavior to display a certain set of columns,
% change the stats field when running markEye as follows:
% > markEye('my.eye','stats','x y pd status');
%
% To see a (currently cryptic) list of legal stats, read the
% export.mconf file.  It has a list of possible fields for 'stats' in the first
% column in braces.  More helpful documentation is forthcoming.
% 
% To see other settings to change, edit defaultparams.m or type 
% > defaultparams
% to see.
%
% MAIN FILES
%*****************************************
% markEye: for marking eye or eyeS files, can be called from the
% MATLAB command line.
% NB: A GUI used to be available for the markEye software but has since
% been discontinued.  We may reopen this GUI as demand arises.
% 
% The syntax is as follows: 
% markEye(<files>,[alias], ... ,[option,option-value],
%...);
%
% files:  The file may be one .eye file or a wildcard path to several
% 	  eye files, i.e. /path/to/*.eye.  
% 	  During batch mode, files are processed in alphabetical order.
%
%	  Note: If only one .eye file is given, markEye will return
%	  a structure containing the eyedata.
%
% alias:  The aliases are listed in parseinputs.m.  They are used
% 	  to succinctly wrap a list of options.  Example aliases can 
% 	  be found in parseinputs.m.  If you have more than one
%	  eyetracking setup, the different configuration information can be
%	  placed here.
%
%	  Advanced users: Additional aliases can be coded to modify
%	  the run-time behavior for each .eye file when running in batch. 
%	  See the add_region alias for an example. 
% 
% option: Options are listed in defaultparams.m with descriptions.
% 	  Units are in the comments.
% 	  Currently there are options for most adjustable parameters
% 	  in the markup functions.  If you want to introduce a 
%	  new option, you must add it to the definitions in 
%	  defaultparams.  
%
% IMPORTANT OPTIONS:
%******************************************
%
% stats: (default for all samples) [x y pupil_diameter status] 
% (default for marked events) [targetx targety t_on t_off t_to_nextevent num_this_event_type]  
% 	  This controls which fields appear in the markEye file output.
%	  A list of allowed stats can be seen in exportEyeData.m.
% 	  The stat header names must be given in a cell array, that is, a
%	  curly-bracketed array of strings.
%	  A list of legal stats can be seen in the first column of 
% 	  export.mconf.  Any field generated in the data structure can 
% 	  technically be used as a statistic.  Use fieldnames on the
% 	  output of a single markEye command to see the possible fields.
%
% stats_out_on_ev:
%
%     This controls which events are given extended output.  
% 
% verbose:
%	  This controls the output of markEye.  Levels are as follows:
%	    -Inf shows nothing ('silent' alias)
% 	    -1 shows only which file is being run ('quiet' alias)
%	    0 shows the general progress for each file (default)
% 	    1 plots the eye trace ('show_trace' alias)
% 	    2 shows output of subroutines
% 	    5 shows data structure
% 	    Inf shows everything ('debug' alias)
%
% sf:	  This is the sampling rate of the eyetracker being used, in
% 	  Hz.
%
% ppd:	  This is the pixels per degree in each dimension (x and y).
%
% autosave:
%         This options allows the user to automatically save each
% 	  trace. Default is yes.
%	  
% GENERAL USAGE
%******************************************
%
% 1) Run markEye on a set of eye files.  
%    These .eye files should be space-delimited with a fixed number of columns.  
%    Individual header data can be written for each .eye file (such as 
%    sampling frequency) as a ## headed line.  
%    The format of these is as follows:
%    ## period 240
%    ## ppd 35
%    These headers can also change the individual options of each eye file.
%    More details are in importEyeData.m.
%
% 2) Check the plots to see how your data is being marked.  The codes
%    have a specific color associated: see plotTrace.m for more.
%
% 3) To control what statistics are sent to ezvision, change the 'stats' 
%    parameter in the command line.  A list of legal stats can be seen
%    in export.mconf. 
%
% 4) To see other possible parameters that can be changed, look in 
%    defaultparams.m.  To see a list of standard options, look in 
%    parseinputs.m.
%
% 5) The e-ceyeS files should be ready for ezvision, if you use our software. :)
%
% EXAMPLES
%*****************************************
% For human subjects at ILAB with the old-eyetrackers, use default.  
%
% Alternate settings for monkeys:
%
%   markeye(<file(s)> ,'sf',1000,'ppd',10); % set sampling rate to
%  		     			   % 1000Hz, 10 pix/deg 
%   markeye(<file(s)> ,'sf',1000,'ppd',13); % set sampling rate to
% 		     			   % 1000Hz, 13 pix/deg 
%
% How to run the simplest version with no angle combination
% and no correcting of fishy areas:
%
%   markEye(<file(s)>,'pro-timethresh',0,'pro-anglethresh',0,...
%   'sac-minamp',0,'sac-window',0,'clean-window',0,'autosave',1);
%
% How to batch process many files quietly and get only the pupil
% diameter and status:
%   
%   markEye(/path/to/*.eye,'quiet','stats',{'pd' 'status'}); 
%
% How to see the progess of each trace as it is processed:
%
%   markEye(/pat/to/*.eye,'show_trace');
