%GRAPHKL graph the Kullback-Liebler (KL) distance between two gamma PDF's
%   [Z,B1,B2] = GRAPHKL(DECAY,BETAFAC,INTERVAL,GRAIN) where
%
%   Z is an NxN matrix of surprise values where N = INTERVAL/GRAIN
%
%   B1 is the asymptotic beta value given the decay term.
%
%   B2 is the asymptotic beta' value given the decay term
%
%   DECAY is the decay term for memory as the eta symbol in syrprise This 
%   a number such that 0 < DECAY < BETAFAC. Traditionally BETAFAC is set to
%   1 so that this number should be a positive decimal. 
%
%   BETAFAC is the update constant on the beta term. This can be seen in
%   the beta update equation:
%
%       beta' = beta*DECAY + BETAFAC
%
%   Traditionally, BETAFAC is set to 1 by default. 
%
%   INTERVAL is the interval from 0.1 to INTERVAL over which we will
%   consider the values of alpha. The interval starts at 0.1 since at 0
%   alpha is undefined. 
%
%   GRAIN this is the granularity of the graph. If you set INTERVAL to 10
%   and GRAIN to 0.1 then Z will be a 100x100 array of surprise values
%   ranging between 0.1 and 10 in increments of 0.1.
%
%   EXAMPLE: [Z,B1,B2] = graphkl(0.7,1,10,0.1)
%
%   This shows the graph where the decay term is 0.7 and the beta update
%   factor is 1. We then graph over the interval of 0.1 to 10 in steps of
%   0.1. The array Z contains the values over the interval while B1 and B2
%   are the asymptotic values of beta and beta' derived from the decay
%   term.
% 
%   See also: runsm, newsm, gamma, psi, eulermasch, digamma, betavalues
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

function [Z,b1,b2] = graphkl(decay,betafac,interval,grain)

% For graphing, we will assume a constant beta based on decay rate
[b1, b2] = betavalues(decay,betafac);
b2       = b2 + betafac;

% Create a matrix to put values in. Thus, we will graph the distribution
% by iterating over a large set of values. 
matsize = (interval/grain) * 2;

% Zero out the matrix
Z = zeros(matsize);

% run over each cell in the matrix and compute the surprise value given the
% input beta and alpha values
fprintf('Computing gamma kl values\n');
x = 1;
for a1 = grain : grain: interval*2 + grain,
    fprintf('.');
    if mod(x,10) == 0
        fprintf('\n');
    end
    y = 1;
    for a2 = grain : grain: interval*2 + grain,
        if a1 ~= 0 && a2 ~= 0
            Z(x,y) = abs(klgamma(a1,a2,b1,b2));
        else
            Z(x,y) = 0;
        end
        y = y + 1;
    end
    x = x + 1;
end

fprintf('Done \n');

% Standard matlab graphing functions
h = surfl(Z);
figure(gcf); 
colormap hsv;
shading interp;
set(h,'EdgeColor','k');
axis([0 matsize 0 matsize 0 max(max(Z))])
xlabel(['\alpha value \cdot ', num2str(grain)],'fontsize',18);
ylabel(['\alpha\prime value \cdot ', num2str(grain)],'fontsize',18);
zlabel('Surprise Value','fontsize',18);
title(['Surprise by Alpha Values with \beta = ', num2str(b1), ' and \beta\prime = ', num2str(b2)],'fontsize',18);
set(gcf,'PaperPositionMode','auto');

% uncomment this line to create a paper printout
% print -dps2;


