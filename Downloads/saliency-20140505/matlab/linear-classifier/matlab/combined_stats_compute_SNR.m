function stats = combined_stats_compute_SNR(mean_in, std_in, frame, feature, class_in, conf)

% EXAMPLE: stats = compute_combined_stats(AVG,NEWFRAME,FEATURE,CLASS)
% or STD_STATS = compute_combined_stats(STATS_H2SV2(:,4),FRAME_H2SV2,FEATURE_H2SV2,CLASS_H2SV2)
% or stats = compute_combined_stats(DIFF_AVG,NEWFRAME,FEATURE,CLASS)

stats = struct('Description','Structure to hold stats data');

BASEERR1 = 1.96;   % .05
BASEERR2 = 2.326;  % .025
BASEERR3 = 2.576;  % .005
BASEERR4 = 3.090;  % .001

frame = frame + 1;
feature_label = conf.feature_label;
feature_num   = conf.feature_num;

stats.new_class = zeros(size(mean_in,1),1);

for i=1:size(mean_in,1) 
    % For testing purposes
    if     class_in(i,1) < conf.hardBound
        stats.new_class(i,1) = 1;
    elseif class_in(i,1) < conf.easyBound
        stats.new_class(i,1) = 2;
    else
        stats.new_class(i,1) = 3;
    end
end

stats.new_feature   = zeros(size(mean_in,1),1);
stats.feature_label = feature_label;
stats.snr           = zeros(size(mean_in,1),1);

for i=1:size(mean_in,1)
    for j=1:feature_num
        if strcmp(feature(i,1),feature_label{j})
            stats.new_feature(i,1) = j;
            break;
        end
    end
end

for i=1:size(mean_in,1)
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

for i=1:size(mean_in,1)
    stats.snr(i,1) = mean_in(i,1)/std_in(i,1);
end

for i=1:size(mean_in,1)
     stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = stats.snr(i,1) + ...
     	stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
     stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))   = 1 + ...          
     	stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
end

stats.mean        = stats.sum .* (1./stats.n);

for i=1:size(mean_in,1)
    stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) + ... 
        (stats.mean(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) - stats.snr(i,1))^2;
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

