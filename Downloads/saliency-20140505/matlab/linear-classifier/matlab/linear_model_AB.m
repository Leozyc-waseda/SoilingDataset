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
function [ldata,tdata,ftdata] = linear_model_AB(conf)

ldata          = struct('Description','Holds the data used in this script');
tdata          = struct('Description','Holds the data two training and testing sets');
ftdata         = struct('Description','Legacy');
% Call to place files into /lab/tmpib/30 if needed
%[unp_status,unp_result] = unix('/lab/mundhenk/linear-classifier/script/unpack.sh');
if strcmp(conf.gatherStats,'yes')
    conf.ABtype = 'stim_AB_Trans-Anims_??_??_???';
    conf = Gather_Stats('AB',conf);
    conf.ABtype = 'stim_AB_Anims-Trans_??_??_???';
    conf = Gather_Stats('AB',conf);
end


% Post process raw data using perl
if strcmp(conf.gatherStats,'yes') || strcmp(conf.testPerl,'yes')
    conf = linear_model_post_process_perl_AB(conf);
end
    
% Do we just want to run surprise only to gather data, but not run the
% matlab analysis?
if ~isfield(conf,'surpriseOnly') || strcmp(conf.surpriseOnly,'no')
  
    % Read in the raw data after perl processing
    [ldata,conf] = linear_model_post_read_data_AB(ldata,conf);
    
    tprint('start');
    [ldata.A,conf] = linear_model_frame_offset_sort_AB(ldata.A,conf);
    [ldata.B,conf] = linear_model_frame_offset_sort_AB(ldata.B,conf);
    tprint('stop');
             
    tprint('start');
    dprint('Getting NEW classes');
    % get Easy / Hard classification
    ldata.A.NEW_CLASS = linear_model_get_new_class(ldata.A.CLASS,conf);
    ldata.B.NEW_CLASS = linear_model_get_new_class(ldata.B.CLASS,conf);
    
    for i =1:ldata.A.OFFSET_CELLS
        ldata.A.OFFCLASS{i}.NEW_CLASS = linear_model_get_new_class(ldata.A.OFFCLASS{i}.CLASS,conf);   
        ldata.B.OFFCLASS{i}.NEW_CLASS = linear_model_get_new_class(ldata.B.OFFCLASS{i}.CLASS,conf);  
        dprint('Compute normalized diff stats');

        ldata.A.OFFCLASS{i}.DIFF_AVG  = linear_model_diff_stats(ldata.A.OFFCLASS{i}.AVG,  ldata.A.OFFCLASS{i}.NEWFRAME);        
        ldata.B.OFFCLASS{i}.DIFF_AVG  = linear_model_diff_stats(ldata.B.OFFCLASS{i}.AVG,  ldata.B.OFFCLASS{i}.NEWFRAME);

        ldata.A.OFFCLASS{i}.DIFF_STD  = linear_model_diff_stats(ldata.A.OFFCLASS{i}.STD,  ldata.A.OFFCLASS{i}.NEWFRAME);
        ldata.B.OFFCLASS{i}.DIFF_STD  = linear_model_diff_stats(ldata.B.OFFCLASS{i}.STD,  ldata.B.OFFCLASS{i}.NEWFRAME);

    end 
    tprint('stop');   
    
    % If requested, graph the class data of the different classes using the
    % basic bar method with sig value. This isn't nessesary for the full
    % script to run.
    if (isfield(conf,'graphABClasses') && strcmp(conf.graphABClasses,'yes'))
        ldata = combined_stats_AB(ldata,conf);
    end
    
    if (isfield(conf,'graphABOffsets') && strcmp(conf.graphABOffsets,'yes'))
        ldata.A = combined_stats_AB_offset(ldata.A,0,'A',conf);
        ldata.B = combined_stats_AB_offset(ldata.B,0,'B',conf);
        for i =1:ldata.A.OFFSET_CELLS
            ldata.A.OFFCLASS{i} = combined_stats_AB_offset(ldata.A.OFFCLASS{i},i,'A',conf);   
            ldata.B.OFFCLASS{i} = combined_stats_AB_offset(ldata.B.OFFCLASS{i},i,'B',conf); 
        end
    end
    conf.endTime = clock;
else
    dprint('NOTICE : Skiping matlab analysis');
end

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


