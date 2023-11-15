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

function result = linear_model_graph_error(stats,conf)
% Example: r = linear_model_graph_error(LINEAR_TEST_STATS,'Max-Chan');

label = conf.typeLabel;

range = [1:size(stats.N,1)]; % conditions
plotN = 1;

filename = [conf.baseDir 'graphs/' conf.condString];

figure('Name',conf.condString,'FileName',filename);

for i=1:4
    fprintf('`%s`\n',stats.idx_label{i});
    subplot(2, 2, plotN);
    xp = errorbar(range, stats.idx(i).mean(:,1), stats.idx(i).bonferror(:,1), '*-r'); hold on;
    hold off;
    legend([xp(1,1)],stats.idx_label{i});
    xlabel('Class Number 1 = Hard 9 = Easy','fontsize',12);
    ylabel([stats.idx_label{i},' Metric: (+/- Bonferroni 95% SE)'],'fontsize',12);
    title([label,' Metric Guess for Hard and Easy Sequences'], 'fontsize',13);
    plotN = plotN + 1;
end

result = 0;