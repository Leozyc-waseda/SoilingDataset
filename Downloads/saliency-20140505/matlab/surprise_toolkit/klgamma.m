%KLGAMMA the Kullback-Liebler (KL) distance between two gamma PDF's
%   D = KLGAMMA(A1,A2,B1,B2) where 
%
%   A1 is the alpha (Shape) hyper parameter in the
%   first gamma probability distribution and A2 is the alpha hyper
%   parameter in the second gamma probability distribution. A2 is also 
%   known as alpha'.    
%
%   B1 is the beta (Rate) hyper parameter (1/theta) in the first gamma probability
%   distribution and B2 is the beta hyperparameter in the second gamma
%   probability distribution. B2 is also known as beta'.  
% 
%   In the general termonology of the kl distance between two gamma
%   functions A1 may be thought of as alpha while A2 may be thought of as
%   alpha prime. This also goes for B1 which is beta and B2 which can be
%   thought of as beta prime. 
%
%   NOTE: A1, B1 and B2 are undefined as 0. All input numbers should be >=
%   0. 
%
%   EQUATION: KL(alpha,alpha',beta,beta'| Gamma PDF) =
%
%             alpha' * log(beta/beta') + log(GAMMA(alpha')/GAMMA(alpha)
%             + beta' * (alpha/beta) + (alpha - alpha') * DIGAMMA(alpha)
%
%   See also: graphkl, runsm, newsm, gamma, psi, eulermasch, digamma,
%   betavalues
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

function d = klgamma(a1,a2,b1,b2)
d = a2 * log(b1/b2) + log(gamma(a2)/gamma(a1)) + b2*(a1/b1) + (a1 - a2) * digamma(a1,1000);