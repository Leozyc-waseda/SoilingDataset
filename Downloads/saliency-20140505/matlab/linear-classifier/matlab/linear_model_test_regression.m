% //////////////////////////////////////////////////////////////////// %
%           Surprise Linear Model - Copyright (C) 2004-2007            %
% by the University of Southern California (USC) and the iLab at USC.  %
% See http://iLab.usc.edu for information about this project.          %
% //////////////////////////////////////////////////////////////////// %
% This file is part of the iLab Neuromorphic Vision Toolkit            %
%                                                                      %
% The Surprise Linear Model is free software; you can                  %
% redistribute it and/or modify it under the terms of the GNU General  %
% Public License as published by the Free Software Foundation; either  %
% version 2 of the License, or (at your option) any later version.     %
%                                                                      %
% The Surprise Linear Model is distributed in the hope                 %
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
% $Revision: 55 $ 
% $Id$
% $HeadURL: https://surprise-mltk.svn.sourceforge.net/svnroot/surprise-mltk/source/surprise_toolkit/example_graph.m $

function stats = linear_model_test_regression(class,stats,conf)

if strcmp(conf.regressionBasic,'yes') || strcmp(conf.graphRegression,'yes')
    stats = basic_regression(class,stats,conf);
end

if strcmp(conf.regressionNorm,'yes') || strcmp(conf.graphRegression,'yes')
    stats = norm_regression(class,stats,conf);
end

if strcmp(conf.regressionBiased,'yes') || strcmp(conf.graphRegression,'yes')
    stats = biased_regression(class,stats,conf);
end

if strcmp(conf.regressionConstrained,'yes') || strcmp(conf.graphRegression,'yes')
    stats = constr_regression(class,stats,conf);
end

% if needed, graph this
if strcmp(conf.graphRegression,'yes')
    X = graph_regression(stats,conf);
end

%=========================================================================
% Basic Linear Regression

function stats = basic_regression(class,stats,conf)

stats.regress   = struct('Description','Holds values from regression testing');

X               = zeros(size(class.difficulty,1)*size(class.difficulty,2),1);
Y               = ones(size(class.difficulty,1)*size(class.difficulty,2),1);

% We do not want training samples in our test sets
if strcmp(conf.regressionUseBound,'yes')
    useBound = 1;
else
    useBound = 0;
end

n = 1;
for i = 1:size(class.difficulty,1)
    for j = 1:size(class.difficulty,2)
        if class.test_class(i,j) >= conf.hardBound && class.test_class(i,j) < conf.easyBound || ~useBound
            X(n,1) = class.test_class(i,j);
            Y(n,1) = class.difficulty(i,j);
            n = n + 1;
        end
    end
end

X = X + 1;

stats.regress.X = [ones(n-1,1) X(1:n-1,1)];
stats.regress.Y = Y(1:n-1,1);

stats.regress = compute_regression(stats.regress,conf,useBound);

%=========================================================================
% Basic Linear Regression

function stats = norm_regression(class,stats,conf)

stats.regnorm   = struct('Description','Holds values from regression testing, normalized version of stats.regress');

X               = zeros(size(class.difficulty,1)*size(class.difficulty,2),1);
Y               = ones(size(class.difficulty,1)*size(class.difficulty,2),1);

% We do not want training samples in our test sets
if strcmp(conf.regressionUseBound,'yes')
    useBound = 1;
else
    useBound = 0;
end

n = 1;
for i = 1:size(class.difficulty,1)
    for j = 1:size(class.difficulty,2)
        if class.test_class(i,j) >= conf.hardBound && class.test_class(i,j) < conf.easyBound || ~useBound
            X(n,1) = class.test_class(i,j);
            Y(n,1) = class.difficulty(i,j);
            n = n + 1;
        end
    end
end

X = X + 1;

stats.regnorm.Y = (Y(1:n-1,1) - min(min(Y(1:n-1,1)))) ./ (max(max(Y(1:n-1,1))) - min(min(Y(1:n-1,1))));
stats.regnorm.X = [ones(n-1,1) X(1:n-1,1)];

stats.regnorm = compute_regression(stats.regnorm,conf,useBound);

%=========================================================================
% Biased Linear Regression - Assume equal class sizes

function stats = biased_regression(class,stats,conf)

stats.biasreg          = struct('Description','Holds values from biased regression testing');

X = [1:1:9]';
Y = stats.idx(3).mean(:,1);

stats.biasreg.X      = ones(size(X,1),2);

stats.biasreg.X(:,2) = X;
stats.biasreg.Y      = Y;

stats.biasreg = compute_regression(stats.biasreg,conf,0);

%=========================================================================
% Biased Linear Regression - Assume equal class sizes

function stats = constr_regression(class,stats,conf)

stats.regconst       = struct('Description','Holds values from biased regression testing');

X                     = [1+conf.hardBound:1:conf.easyBound]';
Y                     = stats.idx(3).mean(1+conf.hardBound:conf.easyBound,1);

stats.regconst.X      = ones(size(X,1),2);

stats.regconst.X(:,2) = X;
stats.regconst.Y      = Y;

stats.regconst = compute_regression(stats.regconst,conf,1);

%=========================================================================
% graph regression

function X = graph_regression(stats,conf)

label = ['Regression ' conf.typeLabel ' ' conf.condString];

filename = [conf.baseDir 'graphs/regression.' conf.condString];

figure('Name',label,'FileName',filename);

subplot(2, 2, 1);

XX = sub_graph_regression(stats.regress,'Basic',label,conf);

subplot(2, 2, 2);

XX = sub_graph_regression(stats.biasreg,'Biased',label,conf);

subplot(2, 2, 3);

XX = sub_graph_regression(stats.regnorm,'Normalized',label,conf);

subplot(2, 2, 4);

XX = sub_graph_regression(stats.regconst,'Constrained',label,conf);

X = 1;


%=========================================================================
% sub-graph regression

function X = sub_graph_regression(reg,myType,label,conf)

hold on
maxX = max(max(reg.X(:,2)));
X = [0 ; maxX];

% Graph Line of Regression
Y = [reg.b(1,1) ; reg.b(1,1) + reg.b(2,1)*maxX];
plot(X,Y,'g');

% Graph regression low bound
Ylow = [reg.bint(1,1) ; reg.bint(1,1) + reg.bint(2,1)*maxX];
plot(X,Ylow,'--r');

% graph regression high bound
Yhigh = [reg.bint(1,2) ; reg.bint(1,2) + reg.bint(2,2)*maxX];
plot(X,Yhigh,'--r');

% graph ideal regression 
plot(reg.ideal.Xgraph,reg.ideal.Ygraph,'--b');

% Graph a scatter of the data
scatter(reg.X(:,2),reg.Y,'filled');
hold off
title([myType ' ' label], 'fontsize',13);

xlabel('Class Number 1 = Hard 9 = Easy','fontsize',12);
    
X = 1;

%=========================================================================
% Compute regression and get stats

function stats = compute_regression(stats,conf,useBound)

global ERROR_STATE;
global COUNT;

% Call matlabs stats toolbox regression function
try
    [stats.b,stats.bint,stats.r,stats.rint,stats.stats] = regress(stats.Y,stats.X);
    ERROR_STATE.regression.isError = 0;
catch
    ERROR_STATE.regression.isError = 1;    
    fprintf('>>>CAUGHT REGRESSION ERROR iter : %d<<<\n',COUNT);
end

if ERROR_STATE.regression.isError  == 0
    % compute some stats over this
    stats.cfit = stats.b(1) + stats.b(2)*stats.X(:,2);
    stats.cerr = stats.cfit - stats.Y;
    stats.ierr = sum(sum(abs(stats.cerr)))/size(stats.cerr,1);
    stats.serr = sum(sum(stats.cerr .* stats.cerr));
    stats.rmse = sqrt(stats.serr/(size(stats.cerr,1) - 1));
    
    if useBound
        End   = conf.easyBound;
        Start = conf.hardBound + 1;    
    else
        End   = 9;
        Start = 1;
    end

    Max = max(max(stats.Y));
    Min = min(min(stats.Y));

    [stats.ideal.Y stats.ideal.b] = compute_ideal_regression(Start,End,Min,Max);

    maxX = max(max(stats.X(:,2)));

    if useBound
        stats.ideal.Xgraph = [conf.hardBound + 1 ; maxX];
        stats.ideal.Ygraph = [Min ; stats.ideal.Y + stats.ideal.b*maxX];
    else
        stats.ideal.Xgraph = [1 ; 9];
        stats.ideal.Ygraph = [Min ; stats.ideal.Y + stats.ideal.b*maxX];
    end 
else
    ERROR_STATE.regression.stats     = stats;
    ERROR_STATE.regression.conf      = conf;
    ERROR_STATE.regression.useBound  = useBound;
    ERROR_STATE.regression.count     = COUNT;
    ERROR_STATE.regression.isError   = 1;
    stats.rmse = 10;
    stats.b    = 0;
end

%=========================================================================
% Compute the ideal line of regression we would like to acheive

function [Y,b] = compute_ideal_regression(Start,End,Min,Max)

b = (Max - Min)/(End - Start);

Y = -1 * (Start * b) + Min;




