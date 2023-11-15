function stats = combined_stats_compute_stats(data, frame, feature, class_in, conf)

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

if(isfield(conf,'singleFeature') && strcmp(conf.singleFeature,'yes'))
    feature_num   = 1;
else
    feature_num   = conf.feature_num;
end

stats.new_class = zeros(size(data,1),1);

for i=1:size(data,1) 
    % For testing purposes
    if     class_in(i,1) < conf.hardBound
        stats.new_class(i,1) = 1;
    elseif class_in(i,1) < conf.easyBound
        stats.new_class(i,1) = 2;
    else
        stats.new_class(i,1) = 3;
    end
end

stats.feature_label = feature_label;

if(isfield(conf,'singleFeature') && strcmp(conf.singleFeature,'yes'))
    stats.new_feature = ones(size(data,1),1);
else
    stats.new_feature = zeros(size(data,1),1);    
    for i=1:size(data,1)
        for j=1:feature_num
            if strcmp(feature(i,1),feature_label{j})
                stats.new_feature(i,1) = j;
                break;
            end
        end
    end
end

for i=1:size(data,1)
    stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))         = 0;
    stats.zero(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))        = 0; 
    stats.zerosum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))     = 0;     
    stats.min(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))         = data(1,1);
    stats.max(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))         = data(1,1);
    stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))           = 0;
    stats.mean(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))        = 0;
    stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))         = 0;
    stats.stderr(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))      = 0;
    stats.upper(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))       = 0;
    stats.lower(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))       = 0;
    stats.bonfcorrect(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = 0;
    stats.bonferror(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))   = 0;
end

for i=1:size(data,1)
    if data(i,1) > 0
        stats.zero(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) =  stats.zero(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) + 1;
    end
    if data(i,1) > stats.max(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))
        stats.max(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) =  data(i,1);
    end
    if data(i,1) < stats.min(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))
        stats.min(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) =  data(i,1);
    end
    stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = data(i,1)   + stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
    stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))   = 1           + stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
end

stats.mean        = stats.sum .* (1./stats.n);

for i=1:size(data,1)
    stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = stats.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) + ... 
        (stats.mean(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) - data(i,1))^2;
    
    stats.zerosum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) = stats.zero(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1)) / ...
        stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));         
end

if(isfield(conf,'runPairedTests') && strcmp(conf.runPairedTests,'yes')) 
    stats.D_P1 = ones(max(max(frame)),max(max(stats.new_feature)));
    stats.D_P2 = ones(max(max(frame)),max(max(stats.new_feature)));
    for i=1:size(data,1)
        if stats.new_class(i,1) == 1
            stats.Ddata(frame(i,1),stats.new_feature(i,1),stats.D_P1(frame(i,1),stats.new_feature(i,1)),1) = data(i,1);
            stats.D_P1(frame(i,1),stats.new_feature(i,1)) = stats.D_P1(frame(i,1),stats.new_feature(i,1)) + 1;
        elseif stats.new_class(i,1) == 3
            stats.Ddata(frame(i,1),stats.new_feature(i,1),stats.D_P2(frame(i,1),stats.new_feature(i,1)),2) = data(i,1);
            stats.D_P2(frame(i,1),stats.new_feature(i,1)) = stats.D_P2(frame(i,1),stats.new_feature(i,1)) + 1;
        end
    end
    if stats.D_P1 ~= stats.D_P2
        error('Cannot perform paired t-test on non-equal size sets %d ne %d',P1,P2);
    end
    % Hayes Pp 339-340
    stats.D       = stats.Ddata(:,:,:,1) - stats.Ddata(:,:,:,2);
    stats.Dsqr    = stats.D.^2;
    stats.Dbar    = sum(stats.D,3)./stats.D_P1;
    Dstd1         = (sum(stats.Dsqr,3)./(stats.D_P1 - 1)); 
    Dstd2     stats.MWflank.n    = ((stats.D_P1.*(stats.Dbar).^2)./(stats.D_P1 - 1));
    stats.Dstd    = sqrt(Dstd1 - Dstd2);
    stats.Dstderr = stats.Dstd./sqrt(stats.D_P1);
    stats.Dt      = stats.Dbar./stats.Dstderr;
end

if(isfield(conf,'runMWTests') && strcmp(conf.runMWTests,'yes'))
    
    stats.MWcenter.sum = zeros(max(max(frame)),max(max(stats.new_class)),max(max(stats.new_feature)));
    stats.MWcenter.n   = zeros(max(max(frame)),max(max(stats.new_class)),max(max(stats.new_feature)));
    stats.MWflank.sum  = zeros(max(max(frame)),max(max(stats.new_class)),max(max(stats.new_feature)));
    stats.MWflank.n    = zeros(max(max(frame)),max(max(stats.new_class)),max(max(stats.new_feature)));
    stats.MWcenter.std = zeros(max(max(frame)),max(max(stats.new_class)),max(max(stats.new_feature)));
    stats.MWflank.std  = zeros(max(max(frame)),max(max(stats.new_class)),max(max(stats.new_feature)));
    
    minFrame = min(min(frame))+1;
    maxFrame = max(max(frame))-1;
    
    % sum and get N
    for i=1:size(data,1)
        SUM = stats.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
        N   = stats.n(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
        
        stats.MWcenter.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))  = data(i,1) + SUM;
        stats.MWcenter.n(  frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))  = 1         + N;
         
        if((frame(i,1) > minFrame) && (frame(i,1) < maxFrame))     
            stats.MWcenter.sum(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))    = data(i,1)   + SUM;
            stats.MWcenter.n(  frame(i,1),stats.new_class(i,1),stats.new_feature(i,1))    = 1           + N;   
            
            stats.MWflank.sum(frame(i,1) - 1,stats.new_class(i,1),stats.new_feature(i,1)) = data(i,1)   + SUM;
            stats.MWflank.n(frame(i,1)   - 1,stats.new_class(i,1),stats.new_feature(i,1)) = 1           + N;   
            
            stats.MWflank.sum(frame(i,1) + 1,stats.new_class(i,1),stats.new_feature(i,1)) = data(i,1)   + SUM;
            stats.MWflank.n(frame(i,1)   + 1,stats.new_class(i,1),stats.new_feature(i,1)) = 1           + N;
        end
    end
    
    % compute Mean
    stats.MWcenter.mean        = stats.MWcenter.sum .* (1./stats.MWcenter.n);
    stats.MWflank.mean         = stats.MWflank.sum  .* (1./stats.MWflank.n);
    
    % compute Standard Dev
    for i=1:size(data,1)    
        
        STD_C  = stats.MWcenter.std(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));
        MEAN_C = stats.MWcenter.mean(frame(i,1),stats.new_class(i,1),stats.new_feature(i,1));              
        stats.MWcenter.std(frame(i,1)     ,stats.new_class(i,1),stats.new_feature(i,1)) = STD_C + (MEAN_C - data(i,1))^2;
        
        if((frame(i,1) > minFrame) && (frame(i,1) < maxFrame)) 
          
            STD_L  = stats.MWflank.std(frame(i,1)  - 1,stats.new_class(i,1),stats.new_feature(i,1));
            MEAN_L = stats.MWflank.mean(frame(i,1) - 1,stats.new_class(i,1),stats.new_feature(i,1));
            STD_R  = stats.MWflank.std(frame(i,1)  + 1,stats.new_class(i,1),stats.new_feature(i,1));
            MEAN_R = stats.MWflank.mean(frame(i,1) + 1,stats.new_class(i,1),stats.new_feature(i,1));
            
            stats.MWflank.std( frame(i,1) + 1 ,stats.new_class(i,1),stats.new_feature(i,1)) = STD_L + (MEAN_L - data(i,1))^2;
            stats.MWflank.std( frame(i,1) - 1 ,stats.new_class(i,1),stats.new_feature(i,1)) = STD_R + (MEAN_R - data(i,1))^2;
        end
    end

    stats.MWcenter.std    = sqrt(stats.MWcenter.std .* (1./(stats.MWcenter.n - 1)));
    stats.MWcenter.stderr = stats.MWcenter.std .* (1./sqrt(stats.MWcenter.n));
    
    stats.MWflank.std     = sqrt(stats.MWflank.std .* (1./(stats.MWflank.n - 1)));
    stats.MWflank.stderr  = stats.MWflank.std .* (1./sqrt(stats.MWflank.n));

    % compute a basic t
    stats.MW_t            = (abs(stats.MWcenter.mean - stats.MWflank.mean)) ./ (sqrt(stats.MWcenter.stderr.^2 + stats.MWflank.stderr.^2));
end

stats.std          = sqrt(stats.std .* (1./(stats.n - 1)));
stats.stderr       = stats.std .* (1./sqrt(stats.n));

% Pooled stderr
if(isfield(conf,'usePooledByFrame') && strcmp(conf.usePooledByFrame,'yes'))
    stats.poolstd      = sqrt((((stats.n(:,1,:) - 1) .* stats.std(:,1,:).^2) + ...
                               ((stats.n(:,3,:) - 1) .* stats.std(:,3,:).^2)) ./ ...
                              (stats.n(:,1,:) + stats.n(:,3,:) - 2));
    stats.poolstderr   = stats.poolstd .* (1./sqrt(stats.n(:,1,:)) + 1./sqrt(stats.n(:,3,:)));
else
    poolSum = 0;
    poolDF  = 0;
    poolDFE = 0;
    for i=1:size(stats.n,1)
        for j=1:size(stats.n,2)
            poolSum = ((stats.n(i,j) - 1) .* stats.std(i,j).^2) + poolSum;
            poolDF  = stats.n(i,j) - 1                          + poolDF;
            pollDFE = 1./sqrt(stats.n(i,j))                     + pollDFE;
        end
    end
    stats.poolstd    = sqrt(poolSum/poolDF);
    stats.poolstderr = stats.poolstd * pollDFE;
end

stats.bonfcorrect1 = BASEERR1 + (BASEERR1^3 + BASEERR1).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect2 = BASEERR2 + (BASEERR2^3 + BASEERR2).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect3 = BASEERR3 + (BASEERR3^3 + BASEERR3).*(1./(4 .* (stats.n - 2)));
stats.bonfcorrect4 = BASEERR4 + (BASEERR4^3 + BASEERR4).*(1./(4 .* (stats.n - 2)));
stats.basecorrect1 = BASEERR1 * ones(size(stats.n,1),size(stats.n,2));
stats.basecorrect2 = BASEERR2 * ones(size(stats.n,1),size(stats.n,2));
stats.basecorrect3 = BASEERR3 * ones(size(stats.n,1),size(stats.n,2));
stats.basecorrect4 = BASEERR4 * ones(size(stats.n,1),size(stats.n,2));
stats.bonferror    = stats.stderr .* stats.bonfcorrect1;
stats.upper        = stats.mean + stats.bonferror;
stats.lower        = stats.mean - stats.bonferror;
stats.upper_unc    = stats.mean + stats.stderr;
stats.lower_unc    = stats.mean - stats.stderr;
stats.frame        = frame;    




