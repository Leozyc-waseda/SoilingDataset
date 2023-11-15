function ldata = combined_stats_AB(ldata,conf)
    
dprint('Computing Combined AB Stats for output');
    
if isfield(conf,'graphCombinedStats') && strcmp( conf.graphCombinedStats,'yes')
    dprint('Graphing combined stats for output');
    
    if isfield(conf,'graphSpaceStats') && strcmp( conf.graphSpaceStats,'yes')
        dprint('Computing Space A');  
        ldata.A.COMBINED_SPACE_STATS = combined_stats_compute_stats(ldata.A.DIFF_SPACE,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);
        result = combined_stats_graph(ldata.A.COMBINED_SPACE_STATS,'Space_A',conf);  
        
        dprint('Computing Space B');  
        ldata.B.COMBINED_SPACE_STATS = combined_stats_compute_stats(ldata.B.DIFF_SPACE,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);
        result = combined_stats_graph(ldata.B.COMBINED_SPACE_STATS,'Space_B',conf);   
    end
           
    if isfield(conf,'graphDiffStats') && strcmp( conf.graphDiffStats,'yes')  
        dprint('Computing Mean A');
        ldata.A.COMBINED_MEAN_STATS  = combined_stats_compute_stats(ldata.A.DIFF_AVG,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);
        dprint('Computing Std A');
        ldata.A.COMBINED_STD_STATS   = combined_stats_compute_stats(ldata.A.DIFF_STD,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);   
        
        result = combined_stats_graph(ldata.A.COMBINED_MEAN_STATS,'Mean_diff_A',conf);
        result = combined_stats_graph(ldata.A.COMBINED_STD_STATS,'Std_diff_A',conf); 
        
        dprint('Computing Mean B');
        ldata.B.COMBINED_MEAN_STATS  = combined_stats_compute_stats(ldata.B.DIFF_AVG,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);
        dprint('Computing Std B');
        ldata.B.COMBINED_STD_STATS   = combined_stats_compute_stats(ldata.B.DIFF_STD,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);   
        
        result = combined_stats_graph(ldata.B.COMBINED_MEAN_STATS,'Mean_diff_B',conf);
        result = combined_stats_graph(ldata.B.COMBINED_STD_STATS,'Std_diff_B',conf);  
    end
        
    if isfield(conf,'graphTargStats') && strcmp( conf.graphTargStats,'yes')  
        dprint('Computing Target Mean');
        ldata.A.COMBINED_MEAN_TARG_STATS  = combined_stats_compute_stats(ldata.A.DIFF_TARG_AVG,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);
        dprint('Computing Target Std');
        ldata.A.COMBINED_STD_TARG_STATS   = combined_stats_compute_stats(ldata.A.DIFF_TARG_STD,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);
        dprint('Computing Target Space');
        ldata.A.COMBINED_SPACE_TARG_STATS = combined_stats_compute_stats(ldata.A.DIFF_TARG_SPACE,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);
        
        result = combined_stats_graph(ldata.A.COMBINED_MEAN_TARG_STATS,'Mean_targ_diff_A',conf);
        result = combined_stats_graph(ldata.A.COMBINED_STD_TARG_STATS,'Std_targ_diff_A',conf); 
        result = combined_stats_graph(ldata.A.COMBINED_SPACE_TARG_STATS,'Space_targ_A',conf);   
        
        dprint('Computing Target Mean');
        ldata.B.COMBINED_MEAN_TARG_STATS  = combined_stats_compute_stats(ldata.B.DIFF_TARG_AVG,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);
        dprint('Computing Target Std');
        ldata.B.COMBINED_STD_TARG_STATS   = combined_stats_compute_stats(ldata.B.DIFF_TARG_STD,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);
        dprint('Computing Target Space');
        ldata.B.COMBINED_SPACE_TARG_STATS = combined_stats_compute_stats(ldata.B.DIFF_TARG_SPACE,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);
        
        result = combined_stats_graph(ldata.B.COMBINED_MEAN_TARG_STATS,'Mean_targ_diff_B',conf);
        result = combined_stats_graph(ldata.B.COMBINED_STD_TARG_STATS,'Std_targ_diff_B',conf); 
        result = combined_stats_graph(ldata.B.COMBINED_SPACE_TARG_STATS,'Space_targ_B',conf);
    end
    
    if isfield(conf,'graphBasicStats') && strcmp( conf.graphBasicStats,'yes')  
        dprint('Computing Mean');
        ldata.A.COMBINED_MEAN_STATS  = combined_stats_compute_stats(ldata.A.AVG,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);
        dprint('Computing Std');
        ldata.A.COMBINED_STD_STATS   = combined_stats_compute_stats(ldata.A.STD,ldata.A.NEWFRAME,ldata.A.FEATURE,ldata.A.CLASS,conf);
        
        result = combined_stats_graph(ldata.A.COMBINED_MEAN_STATS,'Mean_A',conf);
        result = combined_stats_graph(ldata.A.COMBINED_STD_STATS,'Std_A',conf); 
        
        dprint('Computing Mean');
        ldata.B.COMBINED_MEAN_STATS  = combined_stats_compute_stats(ldata.B.AVG,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);
        dprint('Computing Std');
        ldata.B.COMBINED_STD_STATS   = combined_stats_compute_stats(ldata.B.STD,ldata.B.NEWFRAME,ldata.B.FEATURE,ldata.B.CLASS,conf);
        
        result = combined_stats_graph(ldata.B.COMBINED_MEAN_STATS,'Mean_B',conf);
        result = combined_stats_graph(ldata.B.COMBINED_STD_STATS,'Std_B',conf);
 
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