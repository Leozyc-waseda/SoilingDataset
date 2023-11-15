%NEWSM create a new surprise model object
%   SMOD = NEWSM(DECAY);
%   SMOD = NEWSM(DECAY,INITA);
%   SMOD = NEWSM(DECAY,INITA,INITB);
%   SMOD = NEWSM(DECAY,INITA,INITB,INITLAMB);
%   SMOD = NEWSM(DECAY,INITA,INITB,INITLAMB,UPDATEFAC);
%   SMOD = NEWSM(DECAY,INITA,INITB,INITLAMB,UPDATEFAC,DIM);
%
%   SMOD is a surprise modle structure that contains all the current
%   parameters the that surprise model needs in order to run. It is used so
%   that one can create multiple surprise models at a time and control them
%   all individually. 
%
%   DECAY must be specified as the decay rate in the memory term in the
%   surprise model. This is a value 0 < DECAY < 1. The effect of
%   the decay term can be seen in the alpha and beta update equations:
%
%       alpha' = alpha*decay + lambda
%
%       beta'  = beta*decay + updatefac
%
%   INITA is the initial alpha value. It defualts to 1 and need not be set.
%
%   INITB is the initial beta value. It defaults to 1 and need not be set.
%
%   INITLAMB is the initial lambda value. Lambda is the value of the
%   stimulus input to the surprise model. This value defaults to 0.
%
%   UPDATEFAC is the update factor to the beta term. Most likely this
%   should be left as it is, but you may play with it if you want. The
%   default value is 1.
%
%   DIM this defaults to 1. However, this can be any positive integer. If
%   used it sets surprise to use the experimential multi-variant method. 
%   
%   To see how to use this function in the surprise model see the help
%   on runsm which has an example on running the surprise model. 
%
%   See also: runsm, klgamma, graphkl, gamma, psi, digamma, eulermasch 
%
%   References:
%
%      L. Itti, P. Baldi, A Principled Approach to Detecting Surprising 
%      Events in Video, In: Proc. IEEE Conference on Computer Vision and 
%      Pattern Recognition (CVPR), pp. 631-637, Jun 2005. 
%
%      L. Itti, P. Baldi, Bayesian Surprise Attracts Human Attention, In: 
%      Advances in Neural Information Processing Systems, Vol. 19 
%      (NIPS*2005), pp. 1-8, Cambridge, MA:MIT Press,  2006.
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

function smod = newsm(decay,inita,initb,initlamb,updatefac,dim)

% depending on the number of input arguments, set some values to their
% default values. Otherwise accept the input values
if nargin < 6 dim       = 1; end
if nargin < 5 updatefac = 1; end
if nargin < 4 initlamb  = 0; end
if nargin < 3 initb     = 1; end
if nargin < 2 inita     = 1; end
if nargin < 1 
    error('DECAY value must be specified in calling newsm');
end

% check bounds on decay rate
if decay <= 0 || decay >= 1
    error('Error in DECAY value. It must be set 0 < DECAY < 1');
end

% Set up structures and containers for surprise parameters and values. 
smod = struct('Description','Surprise Model Container');
smod.decay     = decay;
smod.updatefac = updatefac; 
smod.smodisset = 1;

% Use standard univariate model
if dim == 1 
    smod.alpha0    = 1;
    smod.alpha1    = inita;
    smod.alpha2    = 1;
    smod.beta1     = initb;
    smod.beta2     = 1;
    smod.xbar1     = 1;
    smod.xbar2     = 1;
    smod.wbar1     = 1;
    smod.wbar2     = 1;
    smod.surprise  = 0;
    smod.epoch     = 0;
    smod.data0     = 1;
    smod.dim       = 1;
    smod.max       = struct('Description','Maximum and upper bound limits on model');
    smod.options   = struct('debug',1,'graph','surprise','setbetamax','no','factordecay','yes','robbins_monro','no','newton_raphson','no');
    % Obtain asymptotic maximum values for beta and beta'
    [smod.max.beta1,smod.max.beta2] = betavalues(decay,updatefac);
    
% Use experimental multi-variate model
else
    mat            = ones(1,dim);
    smod.alpha0    = mat;
    smod.alpha1    = inita;
    smod.alpha2    = mat;
    smod.beta1     = initb;
    smod.beta2     = mat;
    smod.xbar1     = mat;
    smod.xbar2     = mat; 
    smod.wbar1     = mat;
    smod.wbar2     = mat;
    smod.surprise  = mat * 0;
    smod.epoch     = mat * 0;
    smod.data0     = 1;
    smod.dim       = dim;
    smod.max       = struct('Description','Maximum and upper bound limits on model');
    smod.options   = struct('debug',1,'graph','surprise','setbetamax','no','factordecay','yes','robbins_monro','no','newton_raphson','no');
    % Obtain asymptotic maximum values for beta and beta'
    [smod.max.beta1,smod.max.beta2] = betavalues(decay,updatefac);
end
    
    