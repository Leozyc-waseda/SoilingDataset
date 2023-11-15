% ROC: A tool for performing ROC Analysis.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% The iLab Neuromorphic Vision C++ Toolkit - Copyright (C) 2001 by the %%
%% University of Southern California (USC) and the iLab at USC.         %%
%% See http://iLab.usc.edu for information about this project.          %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% Major portions of the iLab Neuromorphic Vision Toolkit are protected %%
%% under the U.S. patent ``Computation of Intrinsic Perceptual Saliency %%
%% in Visual Environments, and Applications'' by Christof Koch and      %%
%% Laurent Itti, California Institute of Technology, 2001 (patent       %%
%% pending; filed July 23, 2001, following provisional applications     %%
%% No. 60/274,674 filed March 8, 2001 and 60/288,724 filed May 4, 2001).%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%% This file is part of the iLab Neuromorphic Vision C++ Toolkit.       %%
%%                                                                      %%
%% The iLab Neuromorphic Vision C++ Toolkit is free software; you can   %%
%% redistribute it and/or modify it under the terms of the GNU General  %%
%% Public License as published by the Free Software Foundation; either  %%
%% version 2 of the License, or (at your option) any later version.     %%
%%                                                                      %%
%% The iLab Neuromorphic Vision C++ Toolkit is distributed in the hope  %%
%% that it will be useful, but WITHOUT ANY WARRANTY; without even the   %%
%% implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      %%
%% PURPOSE.  See the GNU General Public License for more details.       %%
%%                                                                      %%
%% You should have received a copy of the GNU General Public License    %%
%% along with the iLab Neuromorphic Vision C++ Toolkit; if not, write   %%
%% to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   %%
%% Boston, MA 02111-1307 USA.                                           %%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
%%
%% Primary maintainer for this file: Randolph Voorhies <voorhies at usc dot edu>

function [PD, PFA, AUC] = roc(scoreVector, gtVector)

  [sortedScores, sortedScoresIdx] = sort(-scoreVector);
  
  %Re-sort the ground truth vector so we can know for each
  %event (in descending order of score) whether we had a true
  %positive or a false positive
  TruePositives  = gtVector(sortedScoresIdx)==1;
  FalsePositives = gtVector(sortedScoresIdx)==0;
  
  %Find the cumulative sum to show for each threshold point
  %how many true positives and false positives we would have seen
  TruePositives  = cumsum(TruePositives);
  FalsePositives = cumsum(FalsePositives);
  
  %Normalize the results
  PD  = TruePositives  / sum(gtVector == 1);
  PFA = FalsePositives / sum(gtVector == 0);

  %Calculate the AUC
  AUC = trapz(PFA, PD);
end

