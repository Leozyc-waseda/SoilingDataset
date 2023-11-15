%EULERMASCH Euler-Mascheroni Constant 
%   gamma = EULERMASCH(t) returns the Euler-Mascheroni Constant computed
%   out to t expansions along the taylor series. Large values of t increase
%   the accuracy, but require longer to compute. As such one should pick a
%   value that is not to large (maybe 100). 
%
%   The Euler-Mascheroni constant gamma, sometimes also called the Euler 
%   constant (but not to be confused with the constant e==2.718281...) 
%   is defined as the limit of the sequence:
%   for i=1 to infinity: gamma = gamma + (1/i) - ln(1 + 1/i)
%
%   This function is used by the digamma function since as a constant we
%   can then compute the approximation of the digamma using a taylor series
%   as well.
%
%   See also: gamma, psi, digamma, klgamma, betavalues
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

function gamma = eulermasch(t)

if t < 1 || mod(t,1) ~= 0
    error('The value t in Euler-Mascheroni must be a whole number greater than or equal to 1\n');
end

gamma = 0;
for i=1:t
    gamma = gamma + (1/i) - log(1 + (1/i));
end