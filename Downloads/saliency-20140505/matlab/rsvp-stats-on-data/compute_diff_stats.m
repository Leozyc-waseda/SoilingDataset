function stats = compute_diff_stats(data, frame)

% EXAMPLE: stats = compute_combined_stats(AVG,NEWFRAME)
% or STD_STATS = compute_combined_stats(STATS_H2SV2(:,4),FRAME_H2SV2)

stats = zeros(size(data,1),1);

for i=1:size(data,1)
    if frame(i,:) == 0
        stats(i,:) = 0;
        last       = data(i,:);
    else
        stats(i,:) = last - data(i,:);
        last       = data(i,:);
    end
    
end
