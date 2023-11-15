%NEWALPHABETA compute new alpha and beta values for the surprise model
%   SMOD = NEWALPHABETA(DATA,SMOD,OPTIONS);
%
%   SMOD is a surprise modle structure that contains all the current
%   parameters the that surprise model needs in order to run. It is used so
%   that one can create multiple surprise models at a time and control them
%   all individually. 
%
%   DATA is the current sample. This should be a single floating point
%   value.
%
%   SMOD.OPTIONS is the same options structure you pass to runsm. Here it is
%   used to control certain options such as:
%
%       (1) Factor Decay: The beta values are updated differently to
%       account for the decay term and the introduced time uncertanty in
%       the model. 
%
%           beta' = (alpha*decay + xbar) *(beta*decay + 1) / alpha' 
%
%       (2) Constant Beta: Beta is set as a constant based on its
%       asymptotic final value.
%
%           beta' = - 1 /1 - decay 
%
%       (3) Original Beta: Beta is updated based on each time step as in
%       CVPR 05'
%
%           beta' = beta*decay + 1
%
%   See also: runsm, klgamma, graphkl, gamma, psi, digamma, eulermasch 
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
% Primary maintainer for this file: T. Nathan Mundhenk (gamma(a2)/gamma(a1)<mundhenk@usc.edu>
%

function smod = newalphabeta(data,smod)

smod.xbar2  = (data      + smod.xbar1*smod.decay) / (1 + smod.decay);  
smod.wbar2  = (log(data) + smod.xbar1*smod.decay) / (1 + smod.decay);  

if strcmp(smod.options.robbins_monro,'yes') 
    smod.alpha2 = smod.alpha1 * smod.decay + (log(smod.beta2) - digamma(smod.alpha1 * smod.decay,1000) + log(data)) ...
                 / smod.beta1;
elseif strcmp(smod.options.newton_raphson,'yes') 
    log(smod.xbar1/smod.alpha1*smod.decay) 
    smod.wbar1
    digamma(smod.alpha1,1000)
    1/smod.alpha1
    trigamma(smod.alpha1,1000)
    smod.alpha2 = smod.alpha1 * smod.decay + (log(smod.xbar1/smod.alpha1*smod.decay) - smod.wbar1 + digamma(smod.alpha1*smod.decay,1000)) ...
                  / (1/smod.alpha1*smod.decay - trigamma(smod.alpha1*smod.decay,1000));
else
    smod.alpha2 = smod.alpha1 * smod.decay + data;
end

if strcmp(smod.options.factordecay,'yes') 
    if smod.options.debug > 1
        fprintf('Using option `factordecay` in surprise model\n');
    end
    smod.beta2  = (smod.alpha1*smod.decay + smod.xbar2)*(smod.beta1*smod.decay + 1) / smod.alpha2;   
    if smod.options.debug > 1
        fprintf('Values [%f,%f,%f,%f]\n',smod.xbar1,smod.xbar2,smod.beta1,smod.beta2);
    end
    smod.beta1  = smod.beta2;
elseif strcmp(smod.options.setbetamax,'no') 
    if smod.options.debug > 1
        fprintf('Updating beta based on standard model\n');
    end
    smod.beta2 = smod.beta1  * smod.decay + smod.updatefac;
else
    fprintf('Using option `setbetamax` in surprise model\n');
end

smod.xbar1  = smod.xbar2;
smod.wbar1  = smod.wbar2;