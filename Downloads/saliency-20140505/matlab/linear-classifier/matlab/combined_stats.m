function ldata = combined_stats(ldata,conf)
    
dprint('Computing Combined Stats for output');

if isfield(conf,'graphCombinedSumStats') && strcmp( conf.graphCombinedSumStats,'yes')
    fprintf('Graphing combined sum stats for output\n');

    fprintf('Computing Sum Mean\n');
    ldata.COMBINED_MEAN_SUM_STATS  = combined_stats_compute_sum_stats(ldata.AVG,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    fprintf('Computing Sum Std\n');
    ldata.COMBINED_STD_SUM_STATS   = combined_stats_compute_sum_stats(ldata.STD,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    fprintf('Computing Sum Space\n');
    ldata.COMBINED_SPACE_SUM_STATS = combined_stats_compute_sum_stats(ldata.DIFF_SPACE,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    result = combined_stats_graph_sum(ldata.COMBINED_MEAN_SUM_STATS,'Mean',conf);
    result = combined_stats_graph_sum(ldata.COMBINED_STD_SUM_STATS,'Std',conf);
    result = combined_stats_graph_sum(ldata.COMBINED_SPACE_SUM_STATS,'Space',conf);
end

if((isfield(conf,'runMaskSet')        && strcmp(conf.runMaskSet,'yes'))         || ...
   (isfield(conf,'runHardMaskSet')    && strcmp(conf.runHardMaskSet,'yes'))     || ...
   (isfield(conf,'runNotHardMaskSet') && strcmp(conf.runNotHardMaskSet,'yes'))  || ...
   (isfield(conf,'runTransMaskSet')   && strcmp(conf.runTransMaskSet,'yes'))) 
    conf.singleFeature = 'yes'; 
    dprint('Running Mask Set');
    % Simple Ratio    
    dprint('Compute Ratio Pixel');
    ldata.M_RATIOCOUNT        = ldata.M_OVERLAPCOUNT./ldata.M_MASKCOUNT;
    ldata.COMBINED_RATIOCOUNT = combined_stats_compute_stats(ldata.M_RATIOCOUNT,ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf); 
    conf.ThisName             = 'Ratio Pixel Count';
    ldata.RES_RATIOCOUNT      = combined_stats_graph(ldata.COMBINED_RATIOCOUNT,'Ratio Count',conf);       
    ldata.HIST_RATIO_Pre      = combined_stats_graph_hist(ldata.M_RATIOCOUNT, ldata.COMBINED_RATIOCOUNT, 'Ratio Count Pre' , 5 , conf);
    ldata.HIST_RATIO_Targ     = combined_stats_graph_hist(ldata.M_RATIOCOUNT, ldata.COMBINED_RATIOCOUNT, 'Ratio Count Targ', 6 , conf);
    
    % Vinn Ratio
    dprint('Compute Vinn Ratio');
    ldata.M_VINNRATIO         = ldata.M_OVERLAPCOUNT ./ (ldata.M_MASKCOUNT + ldata.M_LAMCOUNT - 2*ldata.M_OVERLAPCOUNT);
    ldata.COMBINED_VINNRATIO  = combined_stats_compute_stats(ldata.M_VINNRATIO,ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf); 
    conf.ThisName             = 'Vinn Ratio';
    ldata.RES_VINNRATIO       = combined_stats_graph(ldata.COMBINED_VINNRATIO,'Vinn Ratio',conf); 
    ldata.HIST_VINN           = combined_stats_graph_hist(ldata.M_VINNRATIO, ldata.COMBINED_VINNRATIO, 'Vinn Ratio', 6 , conf);
    
    % Area Ratio
    dprint('Compute Area Ratio');
    ldata.M_AREARATIO         = ldata.M_OVERLAPCOUNT ./ (ldata.M_MASKCOUNT + ldata.M_LAMCOUNT - ldata.M_OVERLAPCOUNT);
    ldata.COMBINED_AREARATIO  = combined_stats_compute_stats(ldata.M_AREARATIO,ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf); 
    conf.ThisName             = 'Area Ratio';
    ldata.RES_AREARATIO       = combined_stats_graph(ldata.COMBINED_AREARATIO,'Area Ratio',conf); 
    ldata.HIST_AREA           = combined_stats_graph_hist(ldata.M_AREARATIO, ldata.COMBINED_AREARATIO, 'Area Ratio', 6 , conf);
    
    % Mask
    dprint('Compute Mask Count');
    ldata.COMBINED_MASKCOUNT  = combined_stats_compute_stats(ldata.M_MASKCOUNT,ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf); 
    conf.ThisName             = 'Mask Pixel Count';
    ldata.RES_MASKCOUNT       = combined_stats_graph(ldata.COMBINED_MASKCOUNT,'Mask Count',conf);
        
    % Attention Gate/Mask
    dprint('Compute LAM Pixel Count');
    ldata.COMBINED_LAMCOUNT   = combined_stats_compute_stats(ldata.M_LAMCOUNT,ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    conf.ThisName             = 'LAM Pixel Count';
    ldata.RES_LAMCOUNT        = combined_stats_graph(ldata.COMBINED_LAMCOUNT,'LAM Count',conf);   
    ldata.HIST_LAM            = combined_stats_graph_hist(ldata.M_LAMCOUNT, ldata.COMBINED_LAMCOUNT, 'LAM Count', 6 , conf);
        
    % Overlap
    dprint('Compute Overlap Pixel Count');
    ldata.COMBINED_OVERLAPCOUNT = combined_stats_compute_stats(ldata.M_OVERLAPCOUNT,ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    conf.ThisName               = 'Overlap Pixel Count';
    ldata.RES_OVERLAPCOUNT      = combined_stats_graph(ldata.COMBINED_OVERLAPCOUNT,'Overlap Count',conf);
    ldata.HIST_OVERLAP          = combined_stats_graph_hist(ldata.M_OVERLAPCOUNT, ldata.COMBINED_OVERLAPCOUNT, 'Overlap Count', 6 , conf);
     
    conf.singleFeature = 'no';
end

if(isfield(conf,'graphAGStats') && strcmp(conf.graphAGStats,'yes'))
    conf.singleFeature = 'yes';
    % Attention Gate/Mask
    %ldata.LAM_RATIO           = sqrt(ldata.LAM_AVG .* ldata.LAM_STD);
    ldata.LAM_FUNNY_MAX       = combined_stats_compute_funny_max(ldata.LAM_STD, ldata.LAM_AVG, 'max', 'yes');
    ldata.LAM_FUNNY_MIN       = combined_stats_compute_funny_max(ldata.LAM_STD, ldata.LAM_AVG, 'min', 'yes');
    
    ldata.COMBINED_LAMCOUNT   = combined_stats_compute_stats(ldata.M_LAMCOUNT,    ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    ldata.COMBINED_LAM_AVG    = combined_stats_compute_stats(ldata.LAM_AVG,       ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    ldata.COMBINED_LAM_STD    = combined_stats_compute_stats(ldata.LAM_STD,       ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    %ldata.COMBINED_LAM_RATIO  = combined_stats_compute_stats(ldata.LAM_RATIO,     ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    ldata.COMBINED_LAM_FMAX   = combined_stats_compute_stats(ldata.LAM_FUNNY_MAX, ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    ldata.COMBINED_LAM_FMIN   = combined_stats_compute_stats(ldata.LAM_FUNNY_MIN, ldata.M_NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
    
    conf.ThisName             = 'LAM Pixel Count';
    ldata.RES_LAMCOUNT        = combined_stats_graph(ldata.COMBINED_LAMCOUNT,'LAM Count',conf);
    hist                      = combined_stats_graph_chan_hist(ldata.M_LAMCOUNT, ldata.M_NEWFRAME, [], ldata.COMBINED_LAMCOUNT.new_class, ...
                                                               'LAM Count', 5 , 'all-newClass', ldata.RES_LAMCOUNT.t, conf); 
    dprint(['LAM Count Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]); 
                                                           
    ldata.RES_LAM_AVG         = combined_stats_graph(ldata.COMBINED_LAM_AVG,'LAM Avg',conf);
    hist                      = combined_stats_graph_chan_hist(ldata.LAM_AVG, ldata.M_NEWFRAME, [], ldata.COMBINED_LAM_AVG.new_class, ...
                                                               'LAM Avg', 5 , 'all-newClass', ldata.RES_LAM_AVG.t, conf); 
    dprint(['LAM Avg Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]);    
    
    ldata.RES_LAM_STD         = combined_stats_graph(ldata.COMBINED_LAM_STD,'LAM Std',conf);
    hist                      = combined_stats_graph_chan_hist(ldata.LAM_STD, ldata.M_NEWFRAME, [], ldata.COMBINED_LAM_STD.new_class, ...
                                                               'LAM Std', 5 , 'all-newClass', ldata.RES_LAM_STD.t, conf);
    dprint(['LAM Std Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]);  
    
    %result                    = combined_stats_graph(ldata.COMBINED_LAM_RATIO,'LAM Ratio',conf);
    %hist                      = combined_stats_graph_chan_hist(ldata.LAM_RATIO, ldata.M_NEWFRAME, [], ldata.COMBINED_LAM_RATIO.new_class, ...
    %                                                           'LAM Ratio', 5 , 'all-newClass', conf);
    %dprint(['LAM Ratio Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]);
    
    ldata.RES_LAM_FUNNY_MAX   = combined_stats_graph(ldata.COMBINED_LAM_FMAX,'LAM Funny Max',conf);
    hist                      = combined_stats_graph_chan_hist(ldata.LAM_FUNNY_MAX, ldata.M_NEWFRAME, [], ldata.COMBINED_LAM_FMAX.new_class, ...
                                                               'LAM Funny Max', 5 , 'all-newClass', ldata.RES_LAM_FUNNY_MAX.t, conf);
    dprint(['LAM Funny Max Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]);    
    
    ldata.RES_LAM_FUNNY_MIN   = combined_stats_graph(ldata.COMBINED_LAM_FMIN,'LAM Funny Min',conf);
    hist                      = combined_stats_graph_chan_hist(ldata.LAM_FUNNY_MIN, ldata.M_NEWFRAME, [], ldata.COMBINED_LAM_FMIN.new_class, ...
                                                               'LAM Funny Min', 5 , 'all-newClass', ldata.RES_LAM_FUNNY_MIN.t, conf);
    dprint(['LAM Funny Min Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]);
    
    conf.singleFeature = 'no';                                                                                      
end
    
if isfield(conf,'graphCombinedStats') && strcmp( conf.graphCombinedStats,'yes')
    dprint('Graphing combined stats for output');
    
    if isfield(conf,'graphSpaceStats') && strcmp( conf.graphSpaceStats,'yes')
        dprint('Computing Space');  
        ldata.COMBINED_SPACE_STATS = combined_stats_compute_stats(ldata.DIFF_SPACE,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
        result = combined_stats_graph(ldata.COMBINED_SPACE_STATS,'Space',conf);    
    end
    
    if isfield(conf,'graphHistStats') && strcmp( conf.graphHistStats,'yes')
        dprint('Get histogram for mean diff');
        hist = combined_stats_graph_chan_hist(ldata.DIFF_AVG, ldata.NEWFRAME, ldata.FEATURE, ldata.CLASS, ...
                                              'Final Channel Avg', 5, 'final', conf);
                                          
        dprint(['Average Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]);  
        
        hist = combined_stats_graph_chan_hist(ldata.DIFF_STD, ldata.NEWFRAME, ldata.FEATURE, ldata.CLASS, ...
                                              'Final Channel Std', 5, 'final', conf);
                                          
        dprint(['Std Histogram Real Integral ' num2str(hist.rint) ' : Corr ' num2str(hist.corr)]);
    end
       
    if isfield(conf,'graphDiffStats') && strcmp( conf.graphDiffStats,'yes')
        dprint('Computing Mean');
        ldata.COMBINED_MEAN_STATS  = combined_stats_compute_stats(ldata.DIFF_AVG,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
        dprint('Computing Std');
        ldata.COMBINED_STD_STATS   = combined_stats_compute_stats(ldata.DIFF_STD,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);   
        
        result = combined_stats_graph(ldata.COMBINED_MEAN_STATS,'Mean_diff',conf);
        result = combined_stats_graph(ldata.COMBINED_STD_STATS,'Std_diff',conf);  
    end
        
    if isfield(conf,'graphTargStats') && strcmp( conf.graphTargStats,'yes')  
        dprint('Computing Target Mean');
        ldata.COMBINED_MEAN_TARG_STATS  = combined_stats_compute_stats(ldata.DIFF_TARG_AVG,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
        dprint('Computing Target Std');
        ldata.COMBINED_STD_TARG_STATS   = combined_stats_compute_stats(ldata.DIFF_TARG_STD,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
        dprint('Computing Target Space');
        ldata.COMBINED_SPACE_TARG_STATS = combined_stats_compute_stats(ldata.DIFF_TARG_SPACE,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
        
        result = combined_stats_graph(ldata.COMBINED_MEAN_TARG_STATS,'Mean_targ_diff',conf);
        result = combined_stats_graph(ldata.COMBINED_STD_TARG_STATS,'Std_targ_diff',conf); 
        result = combined_stats_graph(ldata.COMBINED_SPACE_TARG_STATS,'Space_targ',conf);
    end
    
    if isfield(conf,'graphBasicStats') && strcmp( conf.graphBasicStats,'yes')  
        dprint('Computing Mean');
        ldata.COMBINED_MEAN_STATS  = combined_stats_compute_stats(ldata.AVG,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
        dprint('Computing Std');
        ldata.COMBINED_STD_STATS   = combined_stats_compute_stats(ldata.STD,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf);
        %fprintf('Computing SNR\n');
        %ldata.COMBINED_SNR         = combined_stats_compute_SNR(ldata.AVG,ldata.STD,ldata.NEWFRAME,ldata.FEATURE,ldata.CLASS,conf); 
        
        result = combined_stats_graph(ldata.COMBINED_MEAN_STATS,'Mean',conf);
        result = combined_stats_graph(ldata.COMBINED_STD_STATS,'Std',conf);
        %result = combined_stats_graph(ldata.COMBINED_SNR,'SNR',conf);  
    end
    
     
end

if isfield(conf,'graphConspicStats') && strcmp( conf.graphConspicStats,'yes')
    dprint('Graphing conspicuity stats for output');

    dprint('Computing Conspic Mean');
    ldata.COMBINED_MEAN_C_STATS  = combined_stats_compute_conspic_stats(ldata.AVG,ldata.NEWFRAME,ldata.FEATURE,conf);
    dprint('Computing Conspic Std');
    ldata.COMBINED_STD_C_STATS   = combined_stats_compute_conspic_stats(ldata.STD,ldata.NEWFRAME,ldata.FEATURE,conf);
    dprint('Computing Conspic Space');
    ldata.COMBINED_SPACE_C_STATS = combined_stats_compute_conspic_stats(ldata.DIFF_SPACE,ldata.NEWFRAME,ldata.FEATURE,conf);

    result = combined_stats_graph_conspic(ldata.COMBINED_MEAN_C_STATS,'Mean',conf);
    result = combined_stats_graph_conspic(ldata.COMBINED_STD_C_STATS,'Std',conf);
    result = combined_stats_graph_conspic(ldata.COMBINED_SPACE_C_STATS,'Space',conf);
end