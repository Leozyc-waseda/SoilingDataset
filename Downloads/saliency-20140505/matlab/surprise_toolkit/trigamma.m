%TRIGAMMA the trigamma function for gamma''(a)/gamma'(a)
%   d = TRIGAMMA(a,k) returns the trigamma function of the number a with the
%   precision of k expansions on a taylor series expansion. It is computed
%   here as:
%
%   for i = 1 to infinity : trigamma = trigamma + 1 / (i + a)^2
%
%   In mathematics, the trigamma function is defined as the logarithmic 
%   derivative of the digamma function:
%
%       \psi(x) =\frac{d}{dx} \ln{\Gamma(x)}= \frac{\Gamma'(x)}{\Gamma(x)}.
%
%   It is the second of the polygamma functions.
% 
%   The value of k can be any whole number >= 1. The larger the value the
%   more accurate the estimate on the taylor expansion estimate, but the
%   more computation is required. Perhaps try a value such as 100. 
%
%   See also: digamma, gamma, psi, eulermasch, klgamma, betavalues
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

function d = trigamma(a,k)

if k < 1 || mod(k,1) ~= 0
    error('The value t in digamma must be a whole number greater than or equal to 1\n');
elseif a == 0
    error('The digamma is only defined for numbers where `a` is not 0\n');
end

if size(a,2) > 1 | size(a,1) > 1
    d   = zeros(size(a,1),size(a,2));
else
    d = 0;
end

for i=1:k
    d = d + 1/((a + i)^2);
end