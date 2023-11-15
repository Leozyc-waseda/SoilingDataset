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

function stats = linear_model_difftarg_space(datax, datay, frame)

% EXAMPLE: stats = compute_combined_stats(AVG,NEWFRAME)
% or STD_STATS = compute_combined_stats(STATS_H2SV2(:,4),FRAME_H2SV2)

dprint('diff targ space');

stats = zeros(size(datax,1),1);

for i=1:size(datax,1)  
    if frame(i,:) == 5
        stats(i-5,:) = sqrt((datax(i,:) - datax(i-5,:))^2 + (datay(i,:) - datay(i-5,:))^2); 
        stats(i-4,:) = sqrt((datax(i,:) - datax(i-4,:))^2 + (datay(i,:) - datay(i-4,:))^2); 
        stats(i-3,:) = sqrt((datax(i,:) - datax(i-3,:))^2 + (datay(i,:) - datay(i-3,:))^2); 
        stats(i-2,:) = sqrt((datax(i,:) - datax(i-2,:))^2 + (datay(i,:) - datay(i-2,:))^2); 
        stats(i-1,:) = sqrt((datax(i,:) - datax(i-1,:))^2 + (datay(i,:) - datay(i-1,:))^2);
        stats(i,:)   = 0;
        stats(i+1,:) = sqrt((datax(i+1,:) - datax(i,:))^2 + (datay(i+1,:) - datay(i,:))^2); 
        stats(i+2,:) = sqrt((datax(i+2,:) - datax(i,:))^2 + (datay(i+2,:) - datay(i,:))^2); 
        stats(i+3,:) = sqrt((datax(i+3,:) - datax(i,:))^2 + (datay(i+3,:) - datay(i,:))^2); 
        stats(i+4,:) = sqrt((datax(i+4,:) - datax(i,:))^2 + (datay(i+4,:) - datay(i,:))^2); 
        stats(i+5,:) = sqrt((datax(i+5,:) - datax(i,:))^2 + (datay(i+5,:) - datay(i,:))^2);   
    end
end



