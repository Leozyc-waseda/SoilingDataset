function result = combined_stats_graph_conspic(stats,label,conf)
% Example: r = graph_combined_data(AVG_DIFF_STATS,'Diff Average');
    
range = [conf.feature_num]; % range of frames around target frame
plotN = 1;

filename = [conf.baseDir 'conspic_bar_graphs.' label];
figname  = ['Cospicuity Graph ' label];

figure('Name',figname,'FileName',filename);

subplot(1, 1, plotN);
%t = (stats.mean(:,1,i) - stats.mean(:,3,i)) ./ (sqrt(stats.stderr(:,1,i).^2 + stats.stderr(:,3,i).^2));

xp = bar([stats.mean(1,:) ; stats.mean(2,:); abs(stats.mean(1,:) - stats.mean(2,:)); (stats.mean(1,:)./stats.mean(2,:)).^2],'group'); 

hold on;
legend(xp,stats.feature_label);

xlabel('Conspicuity per feature','fontsize',12);
%ylabel([stats.feature_label{i},' Surprise: (+/- Bonferroni 95% SE)'],'fontsize',12);
title([label,' Conspicuity Differences'], 'fontsize',13);
plotN = plotN + 1;

hold off;
result = 0;