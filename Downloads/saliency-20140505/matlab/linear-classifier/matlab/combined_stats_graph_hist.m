function histo = combined_stats_graph_hist(data, stats, label, frame, conf)

asize = max(max(stats.n(frame,:,1)));

hdata = zeros(asize,2);

ec = 1;
hc = 1;
for i=1:size(data,1)
    if(stats.frame(i,1) == frame)
    	if(stats.new_class(i,1) == 1)
        	hdata(hc,2) = data(i,1);
            hc = hc + 1;
        elseif(stats.new_class(i,1) == 3)
            hdata(ec,1) = data(i,1);
            ec = ec + 1;
        end
    end
end

filename = [conf.baseDir 'histogram.' conf.condString '.' label];
figname  = ['Histogram Graph ' label];

figure('Name',figname,'FileName',filename);               

[hbins,hcenter] = hist(hdata,6);
hold on
plot(hbins(:,1),'-b');
plot(hbins(:,2),'-r');
hold off

stats.histErrArea  = max(hbins,[],2) - min(hbins,[],2);
stats.histNormArea = max(hbins,[],2);
stats.histErr      = 1 - (sum(stats.histErrArea) / sum(stats.histNormArea));
if(isfield(conf,'printT') && strcmp(conf.printT,'yes'))  
    fprintf('Estimated Histogram Error %f\n',stats.histErr);
end

% For large ranges, round the number.
% For small ranges, give decimal only to 2 places
if max(max(hcenter)) - min(min(hcenter)) > 100
    set(gca,'XTickLabel',round(hcenter));
else
    hcenterStr = num2str(hcenter, '%10.2f');
    set(gca,'XTickLabel',hcenterStr);
end

title([label,' Surprise Histogram for Hard and Easy Sequences'], 'fontsize',13);
xlabel('Overlap between mask and attention gate','fontsize',12);
ylabel('Number of samples per bin','fontsize',12);

h = legend('Easy','Hard');

histo.data   = hdata;
histo.bins   = hbins;
histo.center = hcenter;