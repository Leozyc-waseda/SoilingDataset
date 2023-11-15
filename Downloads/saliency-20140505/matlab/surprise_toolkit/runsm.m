%RUNSM run the surprise model on some data
%   SMOD = RUNSM(x,SMOD);
%   SMOD = RUNSM(DATA,SMOD);
%
%   This function when called will run the surprise model on the data
%   provided. It has two modes of operation which are automatically
%   determined by the input of either a single value x or a vector DATA.
%
%       SMOD = RUNSM(x,SMOD)
%       Single Step: In this mode, x is a single value for lambda in the
%       surprise model. By returning the value SMOD back into the model
%       each time, the model is incremented with each new sample. The
%       surprise value is returned as SMOD.surprise for each call. Note
%       that this value is volitile and will change each time SMOD is fed
%       into the surprise model.
%
%       SMOD = RUNSM(DATA,SMOD)
%       Batch: In batch mode, DATA is provided as a column or row vector with 
%       each element representing a new sample at each time step. Thus, if the
%       input to the model is [1 2 3 4 5] the model will run starting with
%       lambda as 1 at t = 1. 2 is then fed in next as time t = 2 etc...
%       The surprise model will return a column vector in SMOD as
%       SMOD.surprise which is the surprive value obtained at each time
%       step. 
%
%   x is a single floating point that is the next input sample to the model
%   if used it sets the model to single step mode.
%
%   DATA is either a column or row vector of samples to the model each one
%   representing a new sample at each new time step.
%
%   SMOD This is the surprise model you are currently running. runsm is
%   basically stateless and stores all the values it needs in SMOD. This 
%   way multiple surprise models can be created and run at the same time. 
%   To create a new SMOD use the NEWSM funcion.   
%
%   SMOD.OPTIONS these are special options to surprise model outside of the
%   state of how it executes. It is input as a structure when calling the
%   surprise model RUNSM. The current options supported are:
%
%       SMOD.OPTIONS.DEBUG this is the debug level for execution. 
%
%           Debug = 0 the default, runsm will run quiet
%           Debug = 1 runsm will run quiet but will keep a record of all
%           the alpha and beta values it computes along the way. This is
%           most useful in batch mode.
%           Debug = 2 does all that level 1 does plus it outputs debug
%           information to the screen.
%
%       SMOD.OPTIONS.GRAPH this will tell the surprise model what to graph if
%       anyting. Possible values are:
%
%           'none'     - The default, nothing is graphed
%           'surprise' - The surprise values are graphed by epoch
%
%       SMOD.OPTIONS.SETMAXBETA this will set beta to 1 - max and beta' to max
%       where max is the asymptotic maximum value of beta given the initial
%       decay term. 
%
%           'no'  - the default is to update beta at each time step
%           'yes' - keep beta at the asymptotic max value
%
%       SMOD.OPTIONS.FACTORDECAY this will use a beta update that factors
%       in the decay in the model. It allows for beta to float given an
%       infinite run time. It is advised to use this for long runs or runs
%       with many samples. 
%
%           'no'  - the default, use the original beta update
%           'yes' - use the decay factor in the beta update
%
%   EXAMPLE CODE:
%
%   smod    = newsm(0.7);
%   data    = [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7];
%   smod.options.debug = 2;
%   smod    = runsm(data,smod);
%
%   EXAMPLE OUTPUTS:
%
%   This will produce the debug output:
%
%       Running in batch mode. Input is a row vector
%       RUNNING INPUT 1.000000 LOOP 1 ALPHA [1.000000,1.700000] BETA [1.000000,1.700000] SURPRISE VALUE 1.105826
%       RUNNING INPUT 2.000000 LOOP 2 ALPHA [1.700000,3.190000] BETA [1.700000,2.190000] SURPRISE VALUE 2.042862
%       RUNNING INPUT 3.000000 LOOP 3 ALPHA [3.190000,5.233000] BETA [2.190000,2.533000] SURPRISE VALUE 3.557973
%       RUNNING INPUT 4.000000 LOOP 4 ALPHA [5.233000,7.663100] BETA [2.533000,2.773100] SURPRISE VALUE 5.580750
%       RUNNING INPUT 5.000000 LOOP 5 ALPHA [7.663100,10.364170] BETA [2.773100,2.941170] SURPRISE VALUE 7.988808
%       RUNNING INPUT 6.000000 LOOP 6 ALPHA [10.364170,13.254919] BETA [2.941170,3.058819] SURPRISE VALUE 10.671441
%       RUNNING INPUT 7.000000 LOOP 7 ALPHA [13.254919,16.278443] BETA [3.058819,3.141173] SURPRISE VALUE 13.547449
%       RUNNING INPUT 8.000000 LOOP 8 ALPHA [16.278443,19.394910] BETA [3.141173,3.198821] SURPRISE VALUE 16.559365
%       RUNNING INPUT 1.000000 LOOP 9 ALPHA [19.394910,14.576437] BETA [3.198821,3.239175] SURPRISE VALUE 20.045781
%       RUNNING INPUT 2.000000 LOOP 10 ALPHA [14.576437,12.203506] BETA [3.239175,3.267422] SURPRISE VALUE 14.778995
%       RUNNING INPUT 3.000000 LOOP 11 ALPHA [12.203506,11.542454] BETA [3.267422,3.287196] SURPRISE VALUE 12.219705
%       RUNNING INPUT 4.000000 LOOP 12 ALPHA [11.542454,12.079718] BETA [3.287196,3.301037] SURPRISE VALUE 11.558517
%       RUNNING INPUT 5.000000 LOOP 13 ALPHA [12.079718,13.455803] BETA [3.301037,3.310726] SURPRISE VALUE 12.168864
%       RUNNING INPUT 6.000000 LOOP 14 ALPHA [13.455803,15.419062] BETA [3.310726,3.317508] SURPRISE VALUE 13.616819
%       RUNNING INPUT 7.000000 LOOP 15 ALPHA [15.419062,17.793343] BETA [3.317508,3.322256] SURPRISE VALUE 15.628042
%
%   The values obtained can be accessed for instance, the surprise values
%   can be seen by typing the command:
%
%   >> smod.surprise
%
%   ans =
%   
%       1.1058
%       2.0429
%       3.5580
%       5.5808
%       7.9888
%      10.6714
%      13.5474
%      16.5594
%      20.0458
%      14.7790
%      12.2197
%      11.5585
%      12.1689
%      13.6168
%      15.6280
%
%   Debug values can be viewed by accessing the structure using the
%   command:
%
%   >> smod.debugdata.alpha1
%
%   ans =
%   
%       1.0000
%       1.7000
%       3.1900
%       5.2330
%       7.6631
%      10.3642
%      13.2549
%      16.2784
%      19.3949
%      14.5764
%      12.2035
%      11.5425
%      12.0797
%      13.4558
%      15.4191
%
%   NOTES:
%
%   (1) Optionally the beta values can be fixed to the maximum asymptotic
%   value they would achieve given infinite time. This should have the
%   effect that surprise is uniform over all time. This avoids a "waking"
%   phase for the model. It's useful if one wants to assume that the agent
%   is already "alert" at time t = 1.
%
%   (2) Surprise can handel negative inputs. This creates surprise
%   associated with a stimuli inversion. The gammakl of a negative number is
%   complex, but it's magnatude is easily obtained using the abs function
%   in matlab. 
% 
%   See also: newsm, klgamma, graphkl, gamma, psi, eulermasch 
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

function smod = runsm(data,smod)

% We require that smod be reset each run. This keeps values clean
if isfield(smod,'smodisset')
    if smod.smodisset == 0
        fprintf('NOTICE: SMOD has been set once in newsm, but is no longer usable\n');
        fprintf('Most likely you have tried to call runsm in batch mode twice on\n');
        fprintf('the same smod. You must make a new smod each time you call a\n');
        fprintf('batch mode run on your data\n');
        error('SMOD is not valid');
    end
else
    error('You must create SMOD using the function newsm before you call this function');
end

% Use the asymptotic max value for beta
if strcmp(smod.options.setbetamax,'yes')
    smod.beta1    = smod.max.beta1;
    smod.beta2    = smod.max.beta2;
end

% Figure out if we are running sample by sample or in batch mode. If
% running in batch mode, we may need to transpose the vector matrix
if size(data,1) > 1
    if smod.options.debug == 2
        fprintf('Running in batch mode. Input is a column vector\n')
    end
    smod = runsmbatch(data,smod);
elseif size(data,2) > 1  
    if smod.options.debug == 2
        fprintf('Running in batch mode. Input is a row vector\n')
    end
    data = data';
    smod = runsmbatch(data,smod);
else
    if smod.options.debug == 2
        fprintf('Running in single step mode\n')
    end
    smod = runsmsingle(data,smod);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Batch Run Function 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call this to run surprise on a sample vector

function smod = runsmbatch(data,smod)

% Create surprise values output matrix
smod.surprise = zeros(size(data,1),1);

% create some debug information if requested
if smod.options.debug > 0
    smod.debugdata        = struct('Description','Debug values from runsm');
    smod.debugdata.alpha1 = zeros(size(data,1),1);
    smod.debugdata.alpha2 = zeros(size(data,1),1);
    smod.debugdata.beta1  = zeros(size(data,1),1);
    smod.debugdata.beta2  = zeros(size(data,1),1);
end

% For each data item run the surprise model on it. This is the core of the
% batch mode surprise model. We compute new beta and alpha values fromt the
% sample value, then we compute the KL distance between the two Gamma PDF's
% using klgamma. We take the absolute value to support negative data
% values. However, using negative values as inputs may not in fact make
% sense. 
for n = 1:size(data,1)
    smod = newalphabeta(data(n,1),smod);
    smod.surprise(n,1) = abs(klgamma(smod.alpha1,smod.alpha2,smod.beta1,smod.beta2));
    
    if smod.options.debug > 0
        smod.debugdata.alpha1(n,1) = smod.alpha1;
        smod.debugdata.alpha2(n,1) = smod.alpha2;
        smod.debugdata.beta1(n,1)  = smod.beta1;
        smod.debugdata.beta2(n,1)  = smod.beta2;
        if smod.options.debug > 1
            fprintf('RUNNING INPUT %f LOOP %d ALPHA [%f,%f] BETA [%f,%f] SURPRISE VALUE %f\n',data(n,1),n,smod.alpha1,smod.alpha2,smod.beta1,smod.beta2,smod.surprise(n,1));
        end
    end
        
    smod.alpha1 = smod.alpha2; 
    if strcmp(smod.options.setbetamax,'no')
        smod.beta1  = smod.beta2;
    end
    smod.epoch  = smod.epoch + 1;
    smod.smodisset = 0;
end

% Graph the surprise results if requested.
if strcmp(smod.options.graph,'surprise')
    graphtop    = max(max([max(max(real(smod.surprise))) max(max(data))]));
    graphbottom = min(min([min(min(real(smod.surprise))) min(min(data))]));
    topsup      = max(max(real(smod.surprise)));
    topdat      = max(max(data));
    scale = (1:1:size(data,1));
    plot(scale,smod.surprise,'-.',scale,data,'-.');
    text(1,topsup - 1,'Surprise Value','fontsize',18,'HorizontalAlignment','left','BackgroundColor',[0.0 0.0 1.0]);
    text(1,topdat - 1,'Input Value   ','fontsize',18,'HorizontalAlignment','left','BackgroundColor',[0.0 1.0 0.0]);
    xlabel('Time','fontsize',18);
    ylabel('Surprise Value','fontsize',18);
    if strcmp(smod.options.setbetamax,'yes')
        title(['Surprise Values with decay \zeta = ', num2str(smod.decay), ' and update on \beta = ', num2str(smod.updatefac), ...
               ' Using \beta\prime max = ', num2str(smod.beta2)],'fontsize',18);    
    elseif strcmp(smod.options.factordecay,'yes')  
        title(['Surprise Values with decay \zeta = ', num2str(smod.decay), ' and update on \beta = ', num2str(smod.updatefac), ...
               ' Using \beta\prime factor decay = ', num2str(smod.beta2)],'fontsize',18);
    else
        title(['Surprise Values with decay \zeta = ', num2str(smod.decay), ' and update on \beta = ', num2str(smod.updatefac)],'fontsize',18);
    end
    axis([0 size(data,1) graphbottom graphtop])
end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single Run Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Call this to run surprise on one sample at a time. 

function smod = runsmsingle(data,smod,options)
% For each data item run the surprise model on it. This is the core of the
% batch mode surprise model. We compute new beta and alpha values fromt the
% sample value, then we compute the KL distance between the two Gamma PDF's
% using klgamma. We take the absolute value to support negative data
% values. However, using negative values as inputs may not in fact make
% sense. 
smod = newalphabeta(data,smod,options);
smod.surprise = abs(klgamma(smod.alpha1,smod.alpha2,smod.beta1,smod.beta2));
smod.alpha1   = smod.alpha2; 
if strcmp(smod.options.setbetamax,'no')
    smod.beta1    = smod.beta2;
end
smod.epoch    = smod.epoch + 1;

