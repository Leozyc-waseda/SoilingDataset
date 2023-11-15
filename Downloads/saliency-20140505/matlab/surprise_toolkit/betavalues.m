%BETAVALUES compute the asymptotic beta and beta' values given decay term
%   [B1,B2] = BETAVALUES(D,U) Given the surprise model with beta 
%   functions with update factor beta' = decay*beta + update_factor we
%   compute the asymptotic values of beta and beta' given infinite
%   updates. This can be used to estimate the true beta and beta' values in
%   the surprise model given the decay term and the update factor which is
%   typically set to 1.
%
%   It is assumed that the decay term is: 0 < D < 1. Otherwise if its
%   greater than or equal to 1, it's value is unbounded and will grow to
%   infinity. A negative number such as 0 > D > -1 will be bounded, but
%   will make beta decay assumptotically towards zero. 
%   
%   EQUATION: beta = beta' = - update_factor / (decay_rate - 1)
%
%   See also: newsm, runsm, gamma, psi, eulermasch, klgamma, graphkl
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

function [b1,b2] = betavalues(d,u)

% Check bounds on decay, it must be less than 1 and greater than 0
if d <= 0 || d >= 1
    error('D must be bounded such that 0 < D < 1\n');
end

% compute beta and beta' at their asymptotic values
b2 = -u ./ (d - 1);
b1 = b2;
