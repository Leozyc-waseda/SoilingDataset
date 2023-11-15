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

function final_class = linear_model_final_classify(class,conf)
%example: FINAL_CLASS = linear_model_final_classify(NEW_CLASS,'lin_ind')

% use skipFeature to ignore the training feature during testing.
if (strcmp(conf.skipFeature,'yes'))
    class.skipFeature = conf.useFeature;
else
    class.skipFeature = -1;
end

if(strcmp(conf.type,'lin_ind'))
    final_class = lin_ind_linear_model_final_classify(class);    
elseif(strcmp(conf.type,'max_ind'))
    final_class = max_ind_linear_model_final_classify(class); 
elseif(strcmp(conf.type,'max_chan'))
    final_class = max_chan_linear_model_final_classify(class); 
else
    error('Invalid Class type specified');
end
   
%=========================================================================

function final_class = max_ind_linear_model_final_classify(class) 
 
final_class = struct('Description','Structure to hold final classification data');   

class.P_Hard_Mean = class.P_Hard_Mean/max(max(max(max(class.P_Hard_Mean))));
class.P_Easy_Mean = class.P_Easy_Mean/max(max(max(max(class.P_Easy_Mean))));
class.P_Hard_Std  = class.P_Hard_Std /max(max(max(max(class.P_Hard_Std))));
class.P_Easy_Std  = class.P_Easy_Std /max(max(max(max(class.P_Easy_Std))));  
    
for i=1:size(class.P_Hard_Mean,1)
    for j=1:size(class.P_Hard_Mean,2)  
        final_class.p_hard(i,j,1)         = (max(max(class.P_Hard_Mean(i,j,5:6,:)))+max(max(class.P_Hard_Std(i,j,5:6,:))));
        final_class.p_easy(i,j,1)         = (max(max(class.P_Easy_Mean(i,j,5:6,:)))+max(max(class.P_Easy_Std(i,j,5:6,:))));
        final_class.difficulty(i,j,1)     = final_class.p_easy(i,j,1) - final_class.p_hard(i,j,1); 
        final_class.new_test_class(i,j,1) = class.new_class(i,j,1,1);
        final_class.test_class(i,j,1)     = class.class(i,j,1,1);  
    end
end

%=========================================================================

function final_class = lin_ind_linear_model_final_classify(class)    
   
final_class = struct('Description','Structure to hold final classification data'); 

class.P_Hard_Mean   = class.P_Hard_Mean  /max(max(max(max(class.P_Hard_Mean))));
class.P_Easy_Mean   = class.P_Easy_Mean  /max(max(max(max(class.P_Easy_Mean))));
class.P_Hard_Std    = class.P_Hard_Std   /max(max(max(max(class.P_Hard_Std))));
class.P_Easy_Std    = class.P_Easy_Std   /max(max(max(max(class.P_Easy_Std))));
class.P_Hard_Space  = class.P_Hard_Space /max(max(max(max(class.P_Hard_Space))));
class.P_Easy_Space  = class.P_Easy_Space /max(max(max(max(class.P_Easy_Space))));

for i=1:size(class.P_Hard_Mean,1)
    for j=1:size(class.P_Hard_Mean,2)  
        sample_hard = 0;
        sample_easy = 0; 
        joint_sample_hard = 0;
        joint_sample_easy = 0;
        
        % independant probability
        for k=1:size(class.P_Hard_Mean,3)      
            for l=1:size(class.P_Hard_Mean,4)
                if isfinite(class.P_Hard_Mean(i,j,k,l)) && (l ~= class.skipFeature)
                    sample_hard = sample_hard + class.P_Hard_Mean(i,j,k,l) + class.P_Hard_Std(i,j,k,l) + class.P_Hard_Space(i,j,k,l);
                    sample_easy = sample_easy + class.P_Easy_Mean(i,j,k,l) + class.P_Easy_Std(i,j,k,l) + class.P_Easy_Space(i,j,k,l);   
                end
            end
        end
    
        % Joint Probability
        for k=1:size(class.P_Hard_Mean,3)  
            for x=k:size(class.P_Hard_Mean,3)
                if(x ~= class.skipFeature)
                    for l=1:size(class.P_Hard_Mean,4) 
                        if isfinite(class.P_Hard_Mean(i,j,k,l)) && (l ~= class.skipFeature)
                            for y=l:size(class.P_Hard_Mean,4)     
                                if isfinite(class.P_Hard_Mean(i,j,x,y))
                                    joint_sample_hard = joint_sample_hard + class.P_Hard_Mean(i,j,k,l)*class.P_Hard_Mean(i,j,x,y) + ...
                                                                            class.P_Hard_Std(i,j,k,l)*class.P_Hard_Std(i,j,x,y)   + ...
                                                                            class.P_Hard_Space(i,j,k,l)*class.P_Hard_Space(i,j,x,y);
                                    joint_sample_easy = joint_sample_easy + class.P_Easy_Mean(i,j,k,l)*class.P_Easy_Mean(i,j,x,y) + ...
                                                                            class.P_Easy_Std(i,j,k,l)*class.P_Easy_Std(i,j,x,y)   + ...
                                                                            class.P_Easy_Space(i,j,k,l)*class.P_Easy_Space(i,j,x,y); 
                                end
                            end
                        end
                    end
                end
            end
        end
    
        joint_sample_hard = 2*sqrt(joint_sample_hard/2);
        joint_sample_easy = 2*sqrt(joint_sample_easy/2);
        
        %joint_sample_hard = 2*sqrt(joint_sample_hard);
        %joint_sample_easy = 2*sqrt(joint_sample_easy);
        
        %sample_hard       = sample_hard;
        %sample_easy       = sample_easy; 
        final_class.sample_hard(i,j,1)       = sample_hard;
        final_class.sample_easy(i,j,1)       = sample_easy;   
        final_class.joint_sample_hard(i,j,1) = joint_sample_hard;
        final_class.joint_sample_easy(i,j,1) = joint_sample_easy;
        final_class.p_hard(i,j,1)            = (sample_hard - joint_sample_hard)/(3*size(class.P_Hard_Mean,3)*size(class.P_Hard_Mean,4));
        final_class.p_easy(i,j,1)            = (sample_easy - joint_sample_easy)/(3*size(class.P_Hard_Mean,3)*size(class.P_Hard_Mean,4));
        
        %final_class.p_hard(i,j,1)            = (sample_hard - joint_sample_hard)/(size(class.P_Hard_Mean,3)*size(class.P_Hard_Mean,4));
        %final_class.p_easy(i,j,1)            = (sample_easy - joint_sample_easy)/(size(class.P_Hard_Mean,3)*size(class.P_Hard_Mean,4));
        
        final_class.difficulty(i,j,1)        = tansig(final_class.p_easy(i,j,1) - final_class.p_hard(i,j,1));
        final_class.jointdiff(i,j,1)         = (joint_sample_easy - joint_sample_hard)/(3*size(class.P_Hard_Mean,3)*size(class.P_Hard_Mean,4));
        final_class.new_test_class(i,j,1)    = class.new_class(i,j,1,1);  
        final_class.test_class(i,j,1)        = class.class(i,j,1,1);
    end
end

%=========================================================================

function final_class = max_chan_linear_model_final_classify(class)    
   
final_class = struct('Description','Structure to hold final classification data'); 

class.P_Hard_Mean = class.P_Hard_Mean/max(max(max(max(class.P_Hard_Mean))));
class.P_Easy_Mean = class.P_Easy_Mean/max(max(max(max(class.P_Easy_Mean))));
class.P_Hard_Std  = class.P_Hard_Std /max(max(max(max(class.P_Hard_Std))));
class.P_Easy_Std  = class.P_Easy_Std /max(max(max(max(class.P_Easy_Std))));

for i=1:size(class.P_Hard_Mean,1)
    for j=1:size(class.P_Hard_Mean,2)  
        sample_hard = 0;
        sample_easy = 0; 
        % independant probability
        for k=1:size(class.P_Hard_Mean,3)   
            if isfinite(class.P_Hard_Mean(i,j,k,:))  
                sample_hard = sample_hard + max(class.P_Hard_Mean(i,j,k,:)) + max(class.P_Hard_Std(i,j,k,:));
                sample_easy = sample_easy + max(class.P_Easy_Mean(i,j,k,:)) + max(class.P_Easy_Std(i,j,k,:));   
            end
        end
            
        final_class.sample_hard(i,j,1)    = sample_hard/(size(class.P_Hard_Mean,3)*2);
        final_class.sample_easy(i,j,1)    = sample_easy/(size(class.P_Hard_Mean,3)*2);   
        final_class.p_hard(i,j,1)         = sample_hard/(size(class.P_Hard_Mean,3)*2);
        final_class.p_easy(i,j,1)         = sample_easy/(size(class.P_Hard_Mean,3)*2); 
        final_class.difficulty(i,j,1)     = tansig(final_class.p_easy(i,j,1) - final_class.p_hard(i,j,1));
        final_class.new_test_class(i,j,1) = class.new_class(i,j,1,1);  
        final_class.test_class(i,j,1)     = class.class(i,j,1,1);
    end
end   