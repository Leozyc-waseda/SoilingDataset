function result = combined_stats_graph_sum(stats,label,conf)

gnum = ceil(conf.feature_num/4);

gbegin = 1;

for g = 1:gnum
    
    range = [-5:5]; % range of frames around target frame
    plotN = 1;

    filename = [conf.baseDir 'combined_graphs_sum.' conf.condString '.' num2str(g) '.' label];
    figname  = ['Combined Graph Sum ' num2str(g) ' ' label];

    figure('Name',figname,'FileName',filename);

    %for i=1:size(stats.mean,3)
    gend = gbegin + 3;
    if(gend > size(stats.mean,2))
        gend = size(stats.mean,2);
    end
    
    for i=gbegin:gend
        fprintf('`%s`\n',stats.feature_label{i});
        subplot(2, 2, plotN);
        %t = (stats.mean(:,1,i) - stats.mean(:,3,i)) ./ (sqrt(stats.stderr(:,1,i).^2 + stats.stderr(:,3,i).^2));
        xp = errorbar(range, stats.mean(:,i), stats.bonferror(:,i), '*-g'); hold on;
        %errorbar(range, stats.mean(:,2,i), stats.bonferror(:,2,i), '*-g'); 
        %yp = errorbar(range, stats.mean(:,3,i), stats.bonferror(:,3,i), '*-b');
        hold off;
        %legend([xp(1,1),yp(1,1)],'Hard','Easy');
        %for j = 1: size(t,1)
        %    if abs(t(j,1)) > max(max([stats.bonfcorrect4(:,1,i) stats.bonfcorrect4(:,3,i)]))
        %        text(j - 6.35, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'****');
        %    elseif abs(t(j,1)) > max(max([stats.bonfcorrect3(:,1,i) stats.bonfcorrect3(:,3,i)]))
        %        text(j - 6.25, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'***');
        %    elseif abs(t(j,1)) > max(max([stats.bonfcorrect2(:,1,i) stats.bonfcorrect2(:,3,i)]))
        %        text(j - 6.15, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'**');
        %    elseif abs(t(j,1)) > max(max([stats.bonfcorrect1(:,1,i) stats.bonfcorrect1(:,3,i)]))
        %        text(j - 6.1, max([stats.mean(j,1,i) + stats.bonferror(j,1,i), stats.mean(j,3,i) + stats.bonferror(j,3,i)]),'*');
        %    end
        %end
        %text(-5,min(min([stats.mean(:,1,i) stats.mean(:,3,i)])),'Easy Sequences','fontsize',12,'HorizontalAlignment','left','Color',[0.0 0.0 1.0]);
        %text(-5,min(min([stats.mean(:,1,i) stats.mean(:,3,i)])) + .025,'Hard Sequences','fontsize',12,'HorizontalAlignment','left','Color',[1.0 0.0 0.0]);  
        if strcmp(label,'Space')
        	axis([-4 6 -1 1]);
        else
            axis([-6 6 -1 1]);
        end
        axis 'auto y';
        xlabel('Frame Number Offset From Target 0','fontsize',12);
        ylabel([stats.feature_label{i},' Surprise (P = 0.05) '],'fontsize',12);
        title([label,' Surprise Value for All Sequences'], 'fontsize',13);
        plotN = plotN + 1;
    end
    gbegin = gbegin + 4;
    print('-dill',[filename '.ai']); % save as illustrator figure
end

result = 0;