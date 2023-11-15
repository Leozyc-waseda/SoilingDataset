function result = graph_combined_data(stats,label)
% Example: r = graph_combined_data(AVG_DIFF_STATS,'Diff Average');

range = [-5:5]; % range of frames around target frame
plotN = 1;

for i=1:size(stats.mean,3)
    if ~strcmp(stats.feature_label{i},'rg') & ~strcmp(stats.feature_label{i},'by') & ~strcmp(stats.feature_label{i},'intensity')
    %else
    %if strcmp(stats.feature_label{i},'h1') | strcmp(stats.feature_label{i},'h2')
        fprintf('`%s`\n',stats.feature_label{i});
        subplot(2, 2, plotN);
        t = (stats.mean(:,1,i) - stats.mean(:,3,i)) ./ (sqrt(stats.stderr(:,1,i).^2 + stats.stderr(:,3,i).^2))
        xp = errorbar(range, stats.mean(:,1,i), stats.bonferror(:,1,i), '*-r'); hold on;
        %errorbar(range, stats.mean(:,2,i), stats.bonferror(:,2,i), '*-g'); 
        yp = errorbar(range, stats.mean(:,3,i), stats.bonferror(:,3,i), '*-b');
        hold off;
        legend([xp(1,1),yp(1,1)],'Hard','Easy');
        for j = 1: size(t,1)
            if abs(t(j,1)) > max(max([stats.bonfcorrect4(:,1,i) stats.bonfcorrect4(:,3,i)]))
                text(j - 6.35, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'****');
            elseif abs(t(j,1)) > max(max([stats.bonfcorrect3(:,1,i) stats.bonfcorrect3(:,3,i)]))
                text(j - 6.25, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'***');
            elseif abs(t(j,1)) > max(max([stats.bonfcorrect2(:,1,i) stats.bonfcorrect2(:,3,i)]))
                text(j - 6.15, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'**');
            elseif abs(t(j,1)) > max(max([stats.bonfcorrect1(:,1,i) stats.bonfcorrect1(:,3,i)]))
                text(j - 6.1, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'*');
            end
        end
        %text(-5,min(min([stats.mean(:,1,i) stats.mean(:,3,i)])),'Easy Sequences','fontsize',12,'HorizontalAlignment','left','Color',[0.0 0.0 1.0]);
        %text(-5,min(min([stats.mean(:,1,i) stats.mean(:,3,i)])) + .025,'Hard Sequences','fontsize',12,'HorizontalAlignment','left','Color',[1.0 0.0 0.0]);
        xlabel('Frame Number Offset From Target 0','fontsize',12);
        ylabel([stats.feature_label{i},' Surprise: (+/- Bonferroni 95% SE)'],'fontsize',12);
        title([label,' Surprise Value for Hard and Easy Sequences'], 'fontsize',13);
        plotN = plotN + 1;
    end
end

result = 0;