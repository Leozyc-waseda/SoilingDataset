function stats_out = condition_combined_data(stats_in,old_frame,mode,alldata,col)

old_mean  = zeros(1,29);
old_count = zeros(1,29);
stats_out = zeros(size(stats_in,1),1);

if mode == 1
    for i=1:size(stats_in,1)
        old_mean(1,old_frame(i,1))  = old_mean(1,old_frame(i,1))  + stats_in(i,1);
        old_count(1,old_frame(i,1)) = old_count(1,old_frame(i,1)) + 1;
    end

    old_mean = old_mean .* (1./old_count)

    for i=1:size(stats_in,1)
        stats_out(i,1) = stats_in(i,1) - old_mean(1,old_frame(i,1));
    end
elseif mode == 2
    min_ch = min(min(stats_in));
    max_ch = max(max(stats_in));
    
    new_mean  = zeros(32,1);
    min_base  = 1000;
    max_base  = 0;
    for i=1:1000
        nmax = max(max(alldata.stats{i}(:,col)));
        nmin = min(min(alldata.stats{i}(:,col)));
        if nmax > max_base
            max_base = nmax;
        end
        if nmin < min_base
            min_base = nmin;
        end
    end
    
    for i=1:1000 
        new_mean = new_mean + (((alldata.stats{i}(:,col) - min_base) ./ ( max_base - min_base)) .* (max_ch - min_ch) + min_ch);
    end
    
    new_mean = new_mean ./ 1000
    
    for i=1:size(stats_in,1)
        stats_out(i,1) = stats_in(i,1) - new_mean(old_frame(i,1),1);
    end
    
else
    error('mode not recognized');
end
