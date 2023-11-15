function stats = combined_stats_compute_conspic_stats(data, frame, feature, conf)

% EXAMPLE: stats = compute_combined_stats(AVG,NEWFRAME,FEATURE,CLASS)
% or STD_STATS = compute_combined_stats(STATS_H2SV2(:,4),FRAME_H2SV2,FEATURE_H2SV2,CLASS_H2SV2)
% or stats = compute_combined_stats(DIFF_AVG,NEWFRAME,FEATURE,CLASS)

stats = struct('Description','Structure to hold stats data');

BASEERR1 = 1.96;   % .05
BASEERR2 = 2.326;  % .025
BASEERR3 = 2.576;  % .005
BASEERR4 = 3.090;  % .001

feature_label = conf.feature_label;
feature_num   = conf.feature_num;

targetFrame = 5; % what frame is the target in?

stats.new_class = zeros(size(data,1),1);

stats.new_feature = zeros(size(data,1),1);
stats.feature_label = feature_label;

for i=1:size(data,1)
    for j=1:feature_num
        if strcmp(feature(i,1),feature_label{j})
            stats.new_feature(i,1) = j;
            break;
        end
    end
end

for i=1:size(data,1)
    stats.sum(1,stats.new_feature(i,1))         = 0;
    stats.n(1,stats.new_feature(i,1))           = 0;
    stats.mean(1,stats.new_feature(i,1))        = 0;
    stats.std(1,stats.new_feature(i,1))         = 0;
    stats.stderr(1,stats.new_feature(i,1))      = 0;
    stats.upper(1,stats.new_feature(i,1))       = 0;
    stats.lower(1,stats.new_feature(i,1))       = 0;
    stats.bonfcorrect(1,stats.new_feature(i,1)) = 0;
    stats.bonferror(1,stats.new_feature(i,1))   = 0; 
    
    stats.sum(2,stats.new_feature(i,1))         = 0;
    stats.n(2,stats.new_feature(i,1))           = 0;
    stats.mean(2,stats.new_feature(i,1))        = 0;
    stats.std(2,stats.new_feature(i,1))         = 0;
    stats.stderr(2,stats.new_feature(i,1))      = 0;
    stats.upper(2,stats.new_feature(i,1))       = 0;
    stats.lower(2,stats.new_feature(i,1))       = 0;
    stats.bonfcorrect(2,stats.new_feature(i,1)) = 0;
    stats.bonferror(2,stats.new_feature(i,1))   = 0;
end

for i=1:size(data,1)
    if frame(i,1) == targetFrame
        stats.sum(1,stats.new_feature(i,1)) = data(i,1)   + stats.sum(1,stats.new_feature(i,1));
        stats.n(1,stats.new_feature(i,1))   = 1           + stats.n(1,stats.new_feature(i,1));
    else
        stats.sum(2,stats.new_feature(i,1))   = data(i,1)   + stats.sum(2,stats.new_feature(i,1));
        stats.n(2,stats.new_feature(i,1))     = 1           + stats.n(2,stats.new_feature(i,1));
    end
end

stats.mean        = stats.sum .* (1./stats.n);

for i=1:size(data,1) 
    if frame(i,1) == targetFrame
        stats.std(1,stats.new_feature(i,1)) = stats.std(1,stats.new_feature(i,1)) + ... 
            (stats.mean(1,stats.new_feature(i,1)) - data(i,1))^2;
    else
    	stats.std(2,stats.new_feature(i,1)) = stats.std(2,stats.new_feature(i,1)) + ... 
            (stats.mean(2,stats.new_feature(i,1)) - data(i,1))^2; 
    end
end

stats.std          = sqrt(stats.std .* (1./(stats.n - 1)));
stats.stderr       = stats.std .* (1./sqrt(stats.n));
stats.bonfcorrect1 = BASEERR1 + (BASEERR1^3 + BASEERR1).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect2 = BASEERR2 + (BASEERR2^3 + BASEERR2).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect3 = BASEERR3 + (BASEERR3^3 + BASEERR3).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect4 = BASEERR4 + (BASEERR4^3 + BASEERR4).*(1./(4 .* (stats.n - 2)));
stats.bonferror    = stats.stderr .* stats.bonfcorrect1;
stats.upper        = stats.mean + stats.bonferror;
stats.lower        = stats.mean - stats.bonferror;

