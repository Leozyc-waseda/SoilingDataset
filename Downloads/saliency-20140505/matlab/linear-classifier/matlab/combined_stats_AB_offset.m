function ldata_AB = combined_stats_AB_offset(ldata_AB,offset,targ,conf)
    
dprint('Computing Offset AB Stats for output');
    
dprint('Computing Mean');
ldata_AB.COMBINED_MEAN_STATS  = combined_stats_compute_stats(ldata_AB.AVG,ldata_AB.NEWFRAME,ldata_AB.FEATURE,ldata_AB.CLASS,conf);
%dprint('Computing Std');
%ldata_AB.COMBINED_STD_STATS   = combined_stats_compute_stats(ldata_AB.DIFF_STD,ldata_AB.NEWFRAME,ldata_AB.FEATURE,ldata_AB.CLASS,conf);
        
conf.specialFeature     = 'yes';
conf.specialFeatureName = 'final-AGmask';
result = combined_stats_graph(ldata_AB.COMBINED_MEAN_STATS,['Mean ' targ ' offset:' num2str(offset)],conf);
%result = combined_stats_graph(ldata_AB.COMBINED_STD_STATS, ['Std '  targ ' offset:' num2str(offset)],conf); 
conf.specialFeature     = 'no';

    
     
