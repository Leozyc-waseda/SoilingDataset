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

function stats = linear_model_analyze(final_class)
% example LINEAR_TEST_STATS = linear_model_analyze(FINAL_CLASS)

BASEERR1 = 1.96;   % .05
BASEERR2 = 2.326;  % .025
BASEERR3 = 2.576;  % .005
BASEERR4 = 3.090;  % .001

stats = struct('Description','Structure to hold final stats data'); 
for i=1:size(final_class.difficulty,1)
    for j=1:size(final_class.difficulty,2)
        stats.mean_hard.sum(final_class.test_class(i,j)+1,1)  = 0;
        stats.mean_easy.sum(final_class.test_class(i,j)+1,1)  = 0;
        stats.mean_diff.sum(final_class.test_class(i,j)+1,1)  = 0;  
        stats.mean_joint.sum(final_class.test_class(i,j)+1,1) = 0; 
        stats.mean_hard.ss(final_class.test_class(i,j)+1,1)   = 0;
        stats.mean_easy.ss(final_class.test_class(i,j)+1,1)   = 0;
        stats.mean_diff.ss(final_class.test_class(i,j)+1,1)   = 0; 
        stats.mean_joint.ss(final_class.test_class(i,j)+1,1)  = 0;
        stats.N(final_class.test_class(i,j)+1,1) = 0;
    end
end
        
for i=1:size(final_class.difficulty,1)
    for j=1:size(final_class.difficulty,2)        
        stats.mean_hard.sum(final_class.test_class(i,j)+1,1)  = stats.mean_hard.sum(final_class.test_class(i,j)+1,1)  + final_class.p_hard(i,j);
        stats.mean_easy.sum(final_class.test_class(i,j)+1,1)  = stats.mean_easy.sum(final_class.test_class(i,j)+1,1)  + final_class.p_easy(i,j);
        stats.mean_diff.sum(final_class.test_class(i,j)+1,1)  = stats.mean_diff.sum(final_class.test_class(i,j)+1,1)  + final_class.difficulty(i,j);
        stats.mean_joint.sum(final_class.test_class(i,j)+1,1) = stats.mean_joint.sum(final_class.test_class(i,j)+1,1) + final_class.jointdiff(i,j);
        stats.N(final_class.test_class(i,j)+1,1) = stats.N(final_class.test_class(i,j)+1,1) + 1;
    end
end


stats.mean_hard.mean  = stats.mean_hard.sum  .* (1./ stats.N);
stats.mean_easy.mean  = stats.mean_easy.sum  .* (1./ stats.N);
stats.mean_diff.mean  = stats.mean_diff.sum  .* (1./ stats.N);
stats.mean_joint.mean = stats.mean_joint.sum .* (1./ stats.N);


for i=1:size(final_class.difficulty,1)
    for j=1:size(final_class.difficulty,2)   
        stats.mean_hard.ss(final_class.test_class(i,j)+1,1)  = (stats.mean_hard.mean(final_class.test_class(i,j)+1,1)  - final_class.p_hard(i,j))^2 + ...
            stats.mean_hard.ss(final_class.test_class(i,j)+1,1);   
        stats.mean_easy.ss(final_class.test_class(i,j)+1,1)  = (stats.mean_easy.mean(final_class.test_class(i,j)+1,1)  - final_class.p_easy(i,j))^2 + ...
            stats.mean_easy.ss(final_class.test_class(i,j)+1,1);   
        stats.mean_diff.ss(final_class.test_class(i,j)+1,1)  = (stats.mean_diff.mean(final_class.test_class(i,j)+1,1)  - final_class.difficulty(i,j))^2 + ...
            stats.mean_diff.ss(final_class.test_class(i,j)+1,1);
        stats.mean_joint.ss(final_class.test_class(i,j)+1,1) = (stats.mean_joint.mean(final_class.test_class(i,j)+1,1) - final_class.jointdiff(i,j))^2 + ...
            stats.mean_joint.ss(final_class.test_class(i,j)+1,1);
    end
end


stats.mean_hard.std  = sqrt(stats.mean_hard.ss  .* (1./(stats.N - 1)));
stats.mean_easy.std  = sqrt(stats.mean_easy.ss  .* (1./(stats.N - 1)));
stats.mean_diff.std  = sqrt(stats.mean_diff.ss  .* (1./(stats.N - 1)));
stats.mean_joint.std = sqrt(stats.mean_joint.ss .* (1./(stats.N - 1)));

stats.mean_hard.stderr       = stats.mean_hard.std  .* (1./sqrt(stats.N));
stats.mean_easy.stderr       = stats.mean_easy.std  .* (1./sqrt(stats.N));
stats.mean_diff.stderr       = stats.mean_diff.std  .* (1./sqrt(stats.N));
stats.mean_joint.stderr      = stats.mean_joint.std .* (1./sqrt(stats.N));

stats.bonfcorrect1 = BASEERR1 + (BASEERR1^3 + BASEERR1).*(1./(4 .* (stats.N - 2)));
stats.bonfcorrect2 = BASEERR2 + (BASEERR2^3 + BASEERR2).*(1./(4 .* (stats.N - 2)));
stats.bonfcorrect3 = BASEERR3 + (BASEERR3^3 + BASEERR3).*(1./(4 .* (stats.N - 2)));
stats.bonfcorrect4 = BASEERR4 + (BASEERR4^3 + BASEERR4).*(1./(4 .* (stats.N - 2)));

stats.mean_hard.bonferror     = stats.mean_hard.stderr .* stats.bonfcorrect1;
stats.mean_hard.upper         = stats.mean_hard.mean + stats.mean_hard.bonferror;
stats.mean_hard.lower         = stats.mean_hard.mean - stats.mean_hard.bonferror;
    
stats.mean_easy.bonferror     = stats.mean_easy.stderr .* stats.bonfcorrect1;
stats.mean_easy.upper         = stats.mean_easy.mean + stats.mean_easy.bonferror;
stats.mean_easy.lower         = stats.mean_easy.mean - stats.mean_easy.bonferror;

stats.mean_diff.bonferror     = stats.mean_diff.stderr .* stats.bonfcorrect1;
stats.mean_diff.upper         = stats.mean_diff.mean + stats.mean_diff.bonferror;
stats.mean_diff.lower         = stats.mean_diff.mean - stats.mean_diff.bonferror;

stats.mean_joint.bonferror    = stats.mean_joint.stderr .* stats.bonfcorrect1;
stats.mean_joint.upper        = stats.mean_joint.mean + stats.mean_joint.bonferror;
stats.mean_joint.lower        = stats.mean_joint.mean - stats.mean_joint.bonferror;   

stats.idx(1) = stats.mean_hard;
stats.idx(2) = stats.mean_easy;
stats.idx(3) = stats.mean_diff;
stats.idx(4) = stats.mean_joint;

stats.idx_label{1} = 'Hard';
stats.idx_label{2} = 'Easy';
stats.idx_label{3} = 'Diff';
stats.idx_label{4} = 'Joint';




