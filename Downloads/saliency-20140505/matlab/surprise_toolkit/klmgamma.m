%KLMGAMMA the Kullback-Liebler (KL) distance between multivariate 
%   gamma PDF's
%   D = KLMGAMMA(A1,A2,B1,B2) where 
%
%   A1 is the alpha (Shape) hyper parameter for independant samples 
%   in the first gamma probability distribution (time 1). This should be a
%   row vector. 
%
%   A2 is the alpha (Shape) hyper parameter for independant samples
%   in the second gamma probability distribution. A2 is also 
%   known as alpha' (time 2). This should be a row vector. 
%
%   B1 is the beta (Rate) hyper parameter (1/theta) in the first gamma probability
%   distribution and B2 is the beta hyperparameter in the second gamma
%   probability distribution. B2 is also known as beta'. This should be a 
%   row vector.  
%
%   NOTE: Most likely B1 = B2 in the McKay bivariate gamma PDF. However,
%   the option is left to the user to try B1 ~= B2. 
% 
%   In the general termonology of the kl distance between two gamma
%   functions AX1 and AY1 may be thought of as alpha while AX2 and AY@ may 
%   be thought of as alpha prime. This also goes for B1 which is beta and 
%   B2 which can be thought of as beta prime. 
%
%   NOTE: A1, A2, B1 and B2 are undefined = 0 and are complex 
%   for values < 0. Thus, all input numbers should be >= 0. 
%
%   EQUATION: The original bivariate KL is derived as:
%             covXY  = alphaX /beta ^2;
%             covXY' = alphaX'/beta'^2;
%             KL(alphaX,alphaX',alphaY,alphaY',cov,cov'| Gamma PDF) =
%
%             ((alphaX' + alphaY') / 2)) * log(alphaX * covXY'/ alphaX'*covXY)
%             +
%             log(GAMMA(alphaX')*GAMMA(alphaY')/GAMMA(alphaX)*GAMMA(alphaY))
%             +
%             (alphaX + alphaY) * sqrt(alphaX'*covXY/alphaX*covXY')
%             -
%             alphaX + (alphaX - alphaX')*DIGAMMA(alphaX)
%             -
%             alphaY + (alphaY - alphaY')*DIGAMMA(alphaY);
%
%             However, this can be simplified and extended to:
%
%             KL(T,T') = SUM(alpha' * LOG(beta/beta')/N) +
%                        PROD(GAMMA(alpha')/GAMMA(alpha)) +
%                        SUM(alpha * (beta'/beta)^(1/N)) +
%                        SUM((-alpha + (alpha-alpha') * DIGAMMA(alpha));  
%
%   NOTES: Hypothetically, all samples in X 
%          should follow the same time scale and as such should have the
%          same beta values. However, we allow the beta values to float in
%          this algorithm. It is up to you whether to fix all beta values
%          to be equal. 
%
%   REFERENCES: Arwini K, Dodson C T J, Felipussi S, Scharcanski J, (2005)
%               Comparing Distance Measures Between Bivariate Gamma
%               Processes. Hyrdrology (IN-PRESS)
%
%   
%   See also: graphkl, runsm, newsm, gamma, psi, eulermasch, digamma,
%   betavalues, klgamma
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

function d = klmgamma(a1,a2,b1,b2)

% Notice that the form of the gamma is very similar to the
% standard KL gamma expressed as:
% d = a2 * log(b1/b2) + log(gamma(a2)/gamma(a1)) + b2*(a1/b1) + (a1 - a2) * digamma(a1,1000);
N     = size(a1,2);
blog  = log(b1 ./ b2) / N;
broot = ((b2 ./ b1)^(1/N)) / N;

d = (1/N) * sum(a2 .* blog)      + ... 
    prod(gamma(a2) ./ gamma(a1)) + ...
    sum(a1 .* broot)             + ...
    sum(-1 * a1 + (a1 - a2) .* digamma(a1));






  
