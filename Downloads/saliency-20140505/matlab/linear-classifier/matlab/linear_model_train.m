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


%Linear model train

% We assume one of two possible states:
% (1) The RSVP sequence is easy
% (2) The RSVP sequence is hard
%
% These are concidered to be mutulally exclusive and the model is predicted
% to be either (decision boundary with probability). Two linear models are
% considered. (1) An independant model where frames are all independant
% without covariance and (2) a covariant model where coveriance is assumed
% between frames.
%
% example model = linear_model_train(DIFF_AVG,DIFF_STD,NEWFRAME,FEATURE,CLASS)
%
% Compair with compute_combined_stats.m

function model = linear_model_train(Mean, Std, Space, frame, feature, class, conf)

UseFeature = conf.useFeature;

stats = struct('Description','Structure to hold stats data');

frame = frame + 1;

stats.new_class = zeros(size(class,1),1);
stats.new_class_label{1} = 'Hard';
stats.new_class_label{2} = 'Medium';
stats.new_class_label{3} = 'Easy';
stats.feature_label      = conf.feature_label;
stats.feature_num        = conf.feature_num;

stats.new_feature = zeros(size(class,1),1);
% re-lable classes into easy,med,hard

dprint(['Labeling classes into easy-medium-hard : Hard ' num2str(conf.hardBound) ' Easy ' num2str(conf.easyBound)]);

stats.new_class = linear_model_get_new_class(class,conf);

%for i=1:size(class,1)
%    if     class(i,1) < conf.hardBound
%        stats.new_class(i,1) = 1;
%    elseif class(i,1) < conf.easyBound
%        stats.new_class(i,1) = 2;
%    else
%        stats.new_class(i,1) = 3;
%    end
%end

for i=1:size(class,1)
    for j=1:stats.feature_num
        if strcmp(feature(i,1),stats.feature_label{j})
            stats.new_feature(i,1) = j;
            break;
        end
    end  
end

for t=1:3

    if t == 1
        dprint('MEAN');
        data = Mean;
    elseif t == 2
        dprint('STD');
        data = Std;
    else
        dprint('SPACE');
        data = Space;
    end
    
    for i=1:size(class,1)
        stats.sum(frame(i,1),stats.new_class(i,1))         = 0;
        stats.n(frame(i,1),stats.new_class(i,1))           = 0;
        stats.mean(frame(i,1),stats.new_class(i,1))        = 0;
        stats.std(frame(i,1),stats.new_class(i,1))         = 0;
    end

    for i=1:size(class,1)
        if stats.new_feature(i,1) == UseFeature
            stats.sum(frame(i,1),stats.new_class(i,1)) = data(i,1)   + stats.sum(frame(i,1),stats.new_class(i,1));
            stats.n(frame(i,1),stats.new_class(i,1))   = 1           + stats.n(frame(i,1),stats.new_class(i,1));
        end
    end

    stats.mean          = stats.sum .* (1./stats.n);

    for i=1:size(class,1)
        if stats.new_feature(i,1) == UseFeature
            stats.std(frame(i,1),stats.new_class(i,1)) = stats.std(frame(i,1),stats.new_class(i,1)) + ... 
                (stats.mean(frame(i,1),stats.new_class(i,1)) - data(i,1))^2;
        end
    end

    stats.std           = sqrt(stats.std .* (1./(stats.n - 1)));

    if t == 1
        model.mean.mean   = stats.mean;
        model.mean.std    = stats.std;
    elseif t == 2
        model.std.mean    = stats.mean;
        model.std.std     = stats.std;
    else
        model.space.mean  = stats.mean;
        model.space.std   = stats.std;
    end
end
    
model.new_class_label   = stats.new_class_label;
model.feature           = stats.feature_label{UseFeature};

