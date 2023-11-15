function stats = compute_combined_stats(data, frame, feature, class)

% EXAMPLE: stats = compute_combined_stats(AVG,NEWFRAME,FEATURE,CLASS)
% or STD_STATS = compute_combined_stats(STATS_H2SV2(:,4),FRAME_H2SV2,FEATURE_H2SV2,CLASS_H2SV2)
% or stats = compute_combined_stats(DIFF_AVG,NEWFRAME,FEATURE,CLASS)

stats = struct('Description','Structure to hold stats data');

BASEERR1 = 1.96;   % .05
BASEERR2 = 2.326;  % .025
BASEERR3 = 2.576;  % .005
BASEERR4 = 3.090;  % .001

frame = frame + 1;

stats.new_class = zeros(size(class,1),1);
stats.new_class_label{1} = 'Hard';
stats.new_class_label{2} = 'Medium';
stats.new_class_label{3} = 'Easy';
stats.feature_label{1}   = 'by';
stats.feature_label{2}   = 'intensity';
stats.feature_label{3}   = 'ori_0';
stats.feature_label{4}   = 'ori_1';
stats.feature_label{5}   = 'ori_2';
stats.feature_label{6}   = 'ori_3';
stats.feature_label{7}   = 'rg';
stats.feature_label{8}   = 'h1';
stats.feature_label{9}   = 'h2';
stats.new_feature = zeros(size(class,1),1);
% re-lable classes into easy,med,hard

for i=1:size(class,1)
    if class(i,1) < 2
        stats.new_class(i,1) = 1;
    elseif class(i,1) < 7
        stats.new_class(i,1) = 2;
    else
        stats.new_class(i,1) = 3;
    end
    
    for j=1:9
        if strcmp(feature(i,1),stats.feature_label{j})
            stats.new_feature(i,1) = j;
            break;
        end
    end
    
end

for i=1:size(class,1)
    stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))         = 0;
    stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))           = 0;
    stats.mean(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))        = 0;
    stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))         = 0;
    stats.stderr(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))      = 0;
    stats.upper(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))       = 0;
    stats.lower(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))       = 0;
    stats.bonfcorrect(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = 0;
    stats.bonferror(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))   = 0;
end

for i=1:size(class,1)
     stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = data(i,1)   + stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
     stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))   = 1           + stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
end

stats.mean        = stats.sum .* (1./stats.n);

for i=1:size(class,1)
    stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) + ... 
        (stats.mean(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) - data(i,1))^2;
end

stats.std          = sqrt(stats.std .* (1./(stats.n - 1)));
stats.stderr       = stats.std .* (1./sqrt(stats.n));
stats.bonfcorrect1 = BASEERR1 + (BASEERR1^3 + BASEERR1).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect2 = BASEERR2 + (BASEERR2^3 + BASEERR2).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect3 = BASEERR3 + (BASEERR3^3 + BASEERR3).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect4 = BASEERR4 + (BASEERR4^3 + BASEERR4).*(1./(4 .* (stats.n - 2)));
stats.bonferror    = stats.stderr .* stats.bonfcorrect1
stats.upper        = stats.mean + stats.bonferror;
stats.lower        = stats.mean - stats.bonferror;

