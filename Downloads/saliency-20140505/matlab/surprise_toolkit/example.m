% This is an example file showing how to use the Bayesian Surprise Model
% Toolkit. In essence you will input bunch os samples each taken at a new
% time step. It is assumed that each sample is taken at a regular interval.
% The return value is the amount of "Surprise" given the stream of data
% given to the model. 
%
% See: L. Itti, P. Baldi, A Principled Approach to Detecting Surprising 
%      Events in Video, In: Proc. IEEE Conference on Computer Vision and 
%      Pattern Recognition (CVPR), pp. 631-637, Jun 2005. 
%
%      L. Itti, P. Baldi, Bayesian Surprise Attracts Human Attention, In: 
%      Advances in Neural Information Processing Systems, Vol. 19 
%      (NIPS*2005), pp. 1-8, Cambridge, MA:MIT Press,  2006.
%
%
%
%   T. Nathan Mundhenk
%   mundhenk@usc.edu
%
% //////////////////////////////////////////////////////////////////// %
% The Baysian Surprise Matlab Toolkit - Copyright (C) 2004-2007        %
% by the University of Southern California (USC) and the iLab at USC.  %
% See http://iLab.usc.edu for information about this project.          %
% //////////////////////////////////////////////////////////////////// %
% This file is part of the Baysian Surprise Matlab Toolkit             %
%                                                                      %
% The Baysian Surprise Matlab Toolkit is free software; you can        %
% redistribute it and/or modify it under the terms of the GNU General  %
% Public License as published by the Free Software Foundation; either  %
% version 2 of the License, or (at your option) any later version.     %
%                                                                      %
% The Baysian Surprise Matlab Toolkit is distributed in the hope       %
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
%

% Create a new surprise model called smod, we can create as many as we want
% but need to give each one its own name. Here we input the decay constant
% which is related to the assumption of the decay of belief over time. 
smod    = newsm(0.7);
% Create (or sample) some data. Each row in the array is a new sample at
% time t+1. 
data    = [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7];
% We set up some to debug, set to 0 so it will not annoy you
smod.options.debug = 2;
% run the model on the data you have. 
smod    = runsm(data,smod);