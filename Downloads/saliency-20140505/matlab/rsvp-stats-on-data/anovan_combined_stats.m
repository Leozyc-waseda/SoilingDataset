function out_stats = anovan_combined_stats(data, newframe, feature, class, sampleNum, frame)

out_stats = struct('description','holds stat data output');

out_stats.feature_label{1}   = 'by';
out_stats.feature_label{2}   = 'intensity';
out_stats.feature_label{3}   = 'ori_0';
out_stats.feature_label{4}   = 'ori_1';
out_stats.feature_label{5}   = 'ori_2';
out_stats.feature_label{6}   = 'ori_3';
out_stats.feature_label{7}   = 'rg';

fullSize = size(data,1);
out_stats.class     = zeros(fullSize/7,7);
out_stats.frame     = zeros(fullSize/7,7);
out_stats.newframe  = zeros(fullSize/7,7);
out_stats.sampleNum = zeros(fullSize/7,7);  
out_stats.data      = zeros(fullSize/7,7);

for j=1:7
    sample{j} = 1;
end

for i=1:fullSize;
    for j=1:7
        if strcmp(out_stats.feature_label{j},feature(i,1))
            out_stats.class(sample{j},j)     = class(i,1);
            out_stats.newframe(sample{j},j)  = newframe(i,1);
            out_stats.sampleNum(sample{j},j) = sampleNum(i,1);
            out_stats.frame(sample{j},j)     = frame(i,1);
            out_stats.data(sample{j},j)      = data(i,1);
            sample{j} = sample{j} + 1;
            break;
        end
    end
end

for j=1:7
    [out_stats.p{j},out_stats.table{j},out_stats.stats{j},out_stats.terms{j}] = anovan(out_stats.data(:,j),  ...
                                                                                { out_stats.newframe(:,j) ;  ...
                                                                                  out_stats.class(:,j) ;     ...
                                                                                  out_stats.frame(:,j) ;     ...  
                                                                                  out_stats.sampleNum(:,j) ; ...
                                                                                  } ,...
                                                                                'full', 3, ...
                                                                                {'New Frame';['Diff Class:', out_stats.feature_label{j} ]; 'Frame ID'; 'Sample Num'} ); 
end

save anovan_stats_avg.mat out_stats  

%[out_stats.p,out_stats.table,out_stats.stats,out_stats.terms] = anovan(out_stats.class(:,1), ...
%                                                                       { out_stats.frame(:,1); ...
%                                                                       out_stats.data(:,1); ...
%                                                                       out_stats.data(:,2); out_stats.data(:,3); out_stats.data(:,4); ...
%                                                                       out_stats.data(:,5); out_stats.data(:,6); out_stats.data(:,7) } ,...
%                                                                       'linear', 3, ...
%                                                                       {'Frame';out_stats.feature_label{1}; out_stats.feature_label{2}; ...
%                                                                       out_stats.feature_label{3} ;out_stats.feature_label{4}; out_stats.feature_label{5}; ...
%                                                                       out_stats.feature_label{6} ;out_stats.feature_label{7} } );


                                                                 