function histo = combined_stats_graph_chan_hist(data, frame, feature, class, label, use_frame, use_feature, t_val, conf)

if nargin == 8
    conf = t_val;
    no_t = 1;
else
    no_t = 0;
end

%asize = init_size;

%hdata = zeros(asize,2);

dprint(['Running channel histogram bound ' num2str(conf.hardBound) ' to ' num2str(conf.easyBound)]); 

if strcmp(use_feature,'all') 
    ec = 1;
    hc = 1;
    for i=1:size(data,1)                          % For all data
        if(frame(i,1) == use_frame)               % For this frame
            if(class(i,1) < conf.hardBound)   
                hdata(hc,1) = data(i,1);
                hc = hc + 1;
            elseif(class(i,1) >= conf.easyBound)
                edata(ec,1) = data(i,1);
                ec = ec + 1;
            end
        end
    end
elseif strcmp(use_feature,'all-newClass')
    ec = 1;
    hc = 1;
    for i=1:size(data,1)                          % For all data
        if(frame(i,1) == use_frame)               % For this frame
            if(class(i,1) == 1)   
                hdata(hc,1) = data(i,1);
                hc = hc + 1;
            elseif(class(i,1) == 3)
                edata(ec,1) = data(i,1);
                ec = ec + 1;
            end
        end
    end
else
    ec = 1;
    hc = 1;
    for i=1:size(data,1)                          % For all data
        if(frame(i,1) == use_frame)               % For this frame
            if(strcmp(feature(i,1),use_feature))  % For this feature
                if(class(i,1) < conf.hardBound)   
                    hdata(hc,1) = data(i,1);
                    hc = hc + 1;
                elseif(class(i,1) >= conf.easyBound)
                    edata(ec,1) = data(i,1);
                    ec = ec + 1;
                end
            end
        end
    end
end

% normalize the sample sizes for display. This creates a better
% picture of the true distributions, but may be statistically dubious.
graph_norm = ec/hc;

dprint(['Number of Samples - Easy ' num2str(ec) ' Hard ' num2str(hc)]);

filename = [conf.baseDir 'histogram.' conf.condString '.' label];
figname  = ['Histogram Graph ' label];

figure('Name',figname,'FileName',filename);               

%[hbins,hcenter] = hist(hdata,20);
%[ebins,ecenter] = hist(edata,20);

[hbins,ebins,center,rint,corr] = hist_norm(hdata,edata,15);

% normalize the "population" so that graphs have similar areas
%hbins = hbins * graph_norm;

hold on
plot(center,ebins,'-b');
plot(center,hbins,'-r');
% plot(center,min(ebins,hbins),':k');
% plot(center,min(ebins,hbins)./max(ebins,hbins),'-g');
hold off

if no_t == 0
    y_int = max(max(ebins,hbins)) - min(min(ebins,hbins));
    x_int = max(max(center))      - min(min(center));
    y_pos = min(min(ebins,hbins)) + y_int/15;
    x_pos = min(min(center))      + x_int/15;

    text(x_pos,y_pos,['T = ' num2str(t_val(use_frame + 1,1)) ' corr = ' num2str(corr)],'FontSize',12);
end

% For large ranges, round the number.
% For small ranges, give decimal only to 2 places
%if max(max(hcenter)) - min(min(hcenter)) > 100
%    set(gca,'XTickLabel',round(hcenter));
%else
%    hcenterStr = num2str(hcenter, '%10.2f');
%    set(gca,'XTickLabel',hcenterStr);
%end

h = legend('Easy','Hard');

title([label,' Surprise Histogram for Hard and Easy Sequences'], 'fontsize',13);
xlabel('Mean Surprise Statistics','fontsize',12);
ylabel('Number of samples per bin','fontsize',12);

print('-dill',[filename '.ai']); % save as illustrator figure

histo.center = center;
histo.rint   = rint;
histo.corr   = corr;


