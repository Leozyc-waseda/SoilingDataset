function result = combined_stats_graph(stats,label,conf)
% Example: r = graph_combined_data(AVG_DIFF_STATS,'Diff Average',conf);

if(isfield(conf,'useBonfCorrect') && strcmp(conf.useBonfCorrect,'yes'))
    errorRange = stats.bonferror;
    pBound1    = stats.bonfcorrect1;
    pBound2    = stats.bonfcorrect2;
    pBound3    = stats.bonfcorrect3;
    pBound4    = stats.bonfcorrect4;
else
    errorRange = stats.stderr * 1.96;  
    pBound1    = stats.basecorrect1;
    pBound2    = stats.basecorrect2;
    pBound3    = stats.basecorrect3;
    pBound4    = stats.basecorrect4;
end

if(isfield(conf,'singleFeature') && strcmp(conf.singleFeature,'yes'))
    gnum = 1;
else
    gnum = ceil(conf.feature_num/4);
end
gbegin = 1;

plotN = 1;

if(isfield(conf,'specialFeature') && strcmp(conf.specialFeature,'yes'))
    filename = [conf.baseDir 'special_graphs.'  conf.condString '.' label]; 
    figname  = ['Combined Graph ' label];
    figure('Name',figname,'FileName',filename);
end

for g = 1:gnum
    
    range = [-5:5]; % range of frames around target frame
    
    if(isfield(conf,'singleFeature') && strcmp(conf.singleFeature,'yes'))
        filename = [conf.baseDir 'single_graphs.'   conf.condString '.' num2str(g) '.' label]; 
    elseif (isfield(conf,'specialFeature') && strcmp(conf.specialFeature,'yes'))
        % do nothing
    else
        filename = [conf.baseDir 'combined_graphs.' conf.condString '.' num2str(g) '.' label];
    end
    
    if(isfield(conf,'runMWTests') && strcmp(conf.runMWTests,'yes'))
    	filename = [filename '.MW'];
        figname  = ['Combined Graph ' num2str(g) ' ' label ' MW'];
    else
        figname  = ['Combined Graph ' num2str(g) ' ' label];
    end
    
    if(isfield(conf,'specialFeature') && strcmp(conf.specialFeature,'yes'))
        % do nothing
    else
        if plotN > 4
            plotN = 1;
        end
    
        if plotN == 1
            figure('Name',figname,'FileName',filename);
        end
    end
    
    %for i=1:size(stats.mean,3)
    if(isfield(conf,'singleFeature') && strcmp(conf.singleFeature,'yes'))
        gend = 1;
    else
        gend = gbegin + 3;
        if(gend > size(stats.mean,3))
            gend = size(stats.mean,3);
        end
    end
    
    for i=gbegin:gend
        if sum(sum(sum(stats.n(:,1,i)))) == 0
            dprint(['Stats Graph SKIPPING zero count ' stats.feature_label{i}]);
        else
            doGraph = 0;
            if(isfield(conf,'specialFeature') && strcmp(conf.specialFeature,'yes'))
                %dprint(['Checking ' conf.specialFeatureName ' ' stats.feature_label{i}]);
                if(strcmp(conf.specialFeatureName,stats.feature_label{i}))
                    dprint(['Found']);
                    doGraph = 1;
                end
            else
                doGraph = 1;
            end
               
            if doGraph == 1
                dprint(['Stats Graph <<<' stats.feature_label{i} '>>> ' conf.condString ' ' num2str(g) ' ' label]);
                if(isfield(conf,'singleFeature') && strcmp(conf.singleFeature,'yes'))
                    subplot(1, 1, plotN);
                elseif(isfield(conf,'specialFeature') && strcmp(conf.specialFeature,'yes'))
                    subplot(1, 1, 1);
                else
                    subplot(2, 2, plotN);
                end

                t           = (stats.mean(:,1,i) - stats.mean(:,3,i)) ./ (sqrt(stats.stderr(:,1,i).^2 + stats.stderr(:,3,i).^2));
                result.err  = 1/2 .* erfc(abs((stats.mean(:,1,i) - stats.mean(:,3,i)) ./ (sqrt(2.*(stats.std(:,1,i).^2 + stats.std(:,3,i).^2)))));
                
                if(isfield(conf,'printT') && strcmp(conf.printT,'yes'))  
                    for n = 1:size(t,1)  
                        if(isfield(conf,'runMWTests') && strcmp(conf.runMWTests,'yes'))
                            fprintf('MW T Test - Hard %f Easy %f\n',stats.MW_t(n,1,i),stats.MW_t(n,3,i)); 
                        else
                            if(isfield(conf,'useBonfCorrect') && strcmp(conf.useBonfCorrect,'yes'))  
                                fprintf('Bonferroni T Test - %f %f %f %f\n',stats.bonfcorrect1(n,1,i),stats.bonfcorrect2(n,1,i),stats.bonfcorrect3(n,1,i),stats.bonfcorrect4(n,1,i));
                            else
                                fprintf('Basic T Test - %f %f %f %f\n',stats.basecorrect1(n,1,i),stats.basecorrect2(n,1,i),stats.basecorrect3(n,1,i),stats.basecorrect4(n,1,i));
                            end
                            for m = 1:size(t,2)
                                fprintf('T: %f ERR: %f\t',abs(t(n,m)),result.err(n,m));
                                fprintf('Hard Mean %f +/- %f Easy Mean %f +/- %f\n',stats.mean(n,1,m),errorRange(n,1,m),stats.mean(n,3,m),errorRange(n,3,m));
                            end
                        end
                    end
                end
                result.t(:,i) = abs(t);
                if(isfield(conf,'runMWTests') && strcmp(conf.runMWTests,'yes'))
                    xp = plot(range, stats.MW_t(:,1,i),'*-r'); hold on;
                    yp = plot(range, stats.MW_t(:,3,i),'*-b');
                else
                    if(isfield(conf,'usePooledStd') && strcmp(conf.usePooledStd,'yes'))
                        if(isfield(conf,'usePooledByFrame') && strcmp(conf.usePooledByFrame,'yes'))
                            xp = errorbar(range, stats.mean(:,1,i), stats.poolstderr(:,i)*1.96, '*-r'); hold on;
                            yp = errorbar(range, stats.mean(:,3,i), stats.poolstderr(:,i)*1.96, '*-b');
                        else
                            xp = errorbar(range, stats.mean(:,1,i), stats.poolstderr*1.96, '*-r'); hold on;
                            yp = errorbar(range, stats.mean(:,3,i), stats.poolstderr*1.96, '*-b');
                        end
                    else
                        xp = errorbar(range, stats.mean(:,1,i), errorRange(:,1,i), '*-r'); hold on;
                        %errorbar(range, stats.mean(:,2,i), errorRange(:,2,i), '*-g'); 
                        yp = errorbar(range, stats.mean(:,3,i), errorRange(:,3,i), '*-b');
                    end
                end
                        
                hold off;
                legend([xp(1,1),yp(1,1)],'Hard','Easy');
                if(isfield(conf,'runMWTests') && strcmp(conf.runMWTests,'yes'))
                else
                    for j = 1: size(t,1)
                        if abs(t(j,1)) > max(max([pBound4(:,1,i) pBound4(:,3,i)]))
                            text(j - 6.35, max([stats.mean(j,1,i) + errorRange(j,1,i), stats.mean(j,3,i) + errorRange(j,3,i)]),'****');
                        elseif abs(t(j,1)) > max(max([pBound3(:,1,i) pBound3(:,3,i)]))
                            text(j - 6.25, max([stats.mean(j,1,i) + errorRange(j,1,i), stats.mean(j,3,i) + errorRange(j,3,i)]),'***');
                        elseif abs(t(j,1)) > max(max([pBound2(:,1,i) pBound2(:,3,i)]))
                            text(j - 6.15, max([stats.mean(j,1,i) + errorRange(j,1,i), stats.mean(j,3,i) + errorRange(j,3,i)]),'**');
                        elseif abs(t(j,1)) > max(max([pBound1(:,1,i) pBound1(:,3,i)]))
                            text(j - 6.1,  max([stats.mean(j,1,i) + errorRange(j,1,i), stats.mean(j,3,i) + errorRange(j,3,i)]),'*');
                        end
                    end
                end
                %text(-5,min(min([stats.mean(:,1,i) stats.mean(:,3,i)])),'Easy Sequences','fontsize',12,'HorizontalAlignment','left','Color',[0.0 0.0 1.0]);
                %text(-5,min(min([stats.mean(:,1,i) stats.mean(:,3,i)])) + .025,'Hard Sequences','fontsize',12,'HorizontalAlignment','left','Color',[1.0 0.0 0.0]);
                if strcmp(label,'Space')
                    axis([-4 6 -1 1]);
                else
                    axis([-6 6 -1 1]);
                end
                axis 'auto y';
                xlabel('Frame Number Offset From Target 0','fontsize',12);

                if(isfield(conf,'singleFeature') && strcmp(conf.singleFeature,'yes'))  
                    if(isfield(conf,'runMWTests') && strcmp(conf.runMWTests,'yes'))
                        ylabel([conf.ThisName,' Surprise: T Divergence'],'fontsize',12);
                    elseif(isfield(conf,'useBonfCorrect') && strcmp(conf.useBonfCorrect,'yes'))
                        ylabel([conf.ThisName,' Surprise: (+/- Bonferroni 95% SE)'],'fontsize',12);  
                    else
                        ylabel([conf.ThisName,' Surprise: (+/- 95% SE)'],'fontsize',12);
                    end
                else
                    if(isfield(conf,'runMWTests') && strcmp(conf.runMWTests,'yes'))
                        ylabel([stats.feature_label{i},' Surprise: T Divergence'],'fontsize',12);
                    elseif(isfield(conf,'useBonfCorrect') && strcmp(conf.useBonfCorrect,'yes'))
                        ylabel([stats.feature_label{i},' Surprise: (+/- Bonferroni 95% SE)'],'fontsize',12); 
                    else
                        ylabel([stats.feature_label{i},' Surprise: (+/- 95% SE)'],'fontsize',12);
                    end
                end
                title([label,' Surprise Value for Hard and Easy Sequences'], 'fontsize',13);
                plotN = plotN + 1;
            end
        end
    end
    gbegin = gbegin + 4;
    print('-dill',[filename '.ai']); % save as illustrator figure
end
