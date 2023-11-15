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

function class = linear_model_classify(model, Mean, Std, Space, frame, tframe, sample, feature, class_in, conf)
%Example: NEW_CLASS = linear_model_classify(model,DIFF_AVG,DIFF_STD,DIFF_SPACE,NEWFRAME,TFRAME,SAMPLE,FEATURE,CLASS,conf)

frame = frame + 1;
%tframe = tframe - 14;
tframe = tframe - 5;
feature_label = conf.feature_label;
feature_num   = conf.feature_num;

new_class = zeros(size(Mean,1),1);

for i=1:size(Mean,1) 
    % For testing purposes
    if     class_in(i,1) < conf.hardBound
        new_class(i,1) = 1;
    elseif class_in(i,1) < conf.easyBound
        new_class(i,1) = 2;
    else
        new_class(i,1) = 3;
    end
end

new_feature = zeros(size(Mean,1),1);

for i=1:size(Mean,1)
    for j=1:feature_num
        if strcmp(feature(i,1),feature_label{j})
            new_feature(i,1) = j;
            break;
        end
    end
end

% Allocate new class data structures
class.new_class    = zeros(max(sample),max(tframe),max(frame),max(new_feature));
class.class        = class.new_class;
class.P_Hard_Mean  = class.new_class;
class.P_Easy_Mean  = class.new_class;
class.P_Hard_Std   = class.new_class;
class.P_Easy_Std   = class.new_class;
class.P_Hard_Space = class.new_class;
class.P_Easy_Space = class.new_class;

% compute p from each sample given mean and std.
for i=1:size(Mean,1) 
     try   
        class.new_class(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))     = new_class(i,1);  
     catch
        fprintf('Not found...\n');
        feature(i,1)
        fprintf('Feature not found input number %d\n',i);
        class.new_class(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))     = new_class(i,1); % include to get std matlab error
        error('Quiting');    
     end
        
     class.class(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))         = class_in(i,1);
     % mean
     class.P_Hard_Mean(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))   = normpdf(Mean(i,1),model.mean.mean(frame(i,1),1),model.mean.std(frame(i,1),1));
     class.P_Easy_Mean(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))   = normpdf(Mean(i,1),model.mean.mean(frame(i,1),3),model.mean.std(frame(i,1),3)); 
     % std
     class.P_Hard_Std(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))    = normpdf(Std(i,1),model.std.mean(frame(i,1),1),model.std.std(frame(i,1),1));
     class.P_Easy_Std(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))    = normpdf(Std(i,1),model.std.mean(frame(i,1),3),model.std.std(frame(i,1),3));  
     % space
     class.P_Hard_Space(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))  = normpdf(Space(i,1),model.space.mean(frame(i,1),1),model.space.std(frame(i,1),1));
     class.P_Easy_Space(sample(i,1),tframe(i,1),frame(i,1),new_feature(i,1))  = normpdf(Space(i,1),model.space.mean(frame(i,1),3),model.space.std(frame(i,1),3));
end

