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

function stats = linear_model_difftarg_stats(data, frame)

% EXAMPLE: stats = compute_combined_stats(AVG,NEWFRAME)
% or STD_STATS = compute_combined_stats(STATS_H2SV2(:,4),FRAME_H2SV2)

dprint(['diff targ - data size ' num2str(size(data,1))]);

stats = zeros(size(data,1),1);

for i=1:size(data,1)
    if frame(i,:) == 5
        stats(i-5,:) = data(i,:) - data(i-5,:);
        stats(i-4,:) = data(i,:) - data(i-4,:);
        stats(i-3,:) = data(i,:) - data(i-3,:);
        stats(i-2,:) = data(i,:) - data(i-2,:);
        stats(i-1,:) = data(i,:) - data(i-1,:);
        stats(i,:)   = 0;
        stats(i+1,:) = data(i,:) - data(i+1,:);
        stats(i+2,:) = data(i,:) - data(i+2,:);   
        stats(i+3,:) = data(i,:) - data(i+3,:);
        stats(i+4,:) = data(i,:) - data(i+4,:);
        stats(i+5,:) = data(i,:) - data(i+5,:);
    end   
end
