function x = dumpErrorRecord(errorRecord,file,dfile,stats,debugLevel)

if debugLevel > 1
    fid = fopen(dfile,'a');
    fprintf(fid,'\n**********************************************\n');
    fprintf(fid,'SIMPLEX DEBUG VALUES\n');
    fprintf(fid,'VALUE %f LOCAL %f ERRTYPE %s ITER %d LOOP %d HOW %s LASTCAT %s\n\n',stats.BEST_VAL,stats.LOC_BEST_VAL,stats.errorType,...
                                                                                      stats.ITERCOUNT,stats.OUTERLOOPS,stats.how,stats.LASTCAT);
    fclose(fid);
end

fid = fopen(file,'a');
fprintf(fid,'VALUE %f LOCAL %f ERRTYPE %s ITER %d LOOP %d HOW %s LASTCAT %s\n',stats.BEST_VAL,stats.LOC_BEST_VAL,stats.errorType,...
                                                                                      stats.ITERCOUNT,stats.OUTERLOOPS,stats.how,stats.LASTCAT);
fprintf(fid,'reflect\t\t\t%f\n',errorRecord.reflect);
y = debugErrDump(errorRecord.reflect,errorRecord.data.reflect,debugLevel,dfile,'reflect',stats.ITERCOUNT,stats.OUTERLOOPS);

if isfield(errorRecord,'expand');
    fprintf(fid,'expand\t\t\t%d:%f\n',errorRecord.time.expand,errorRecord.expand);
    y = debugErrDump(errorRecord.expand,errorRecord.data.expand,debugLevel,dfile,'expand',stats.ITERCOUNT,stats.OUTERLOOPS);
end

if isfield(errorRecord,'contract_outside')
    fprintf(fid,'contract_outside\t%d:%f\n',errorRecord.time.contract_outside,errorRecord.contract_outside);
    y = debugErrDump(errorRecord.contract_outside,errorRecord.data.contract_outside,debugLevel,dfile,'contract_outside',stats.ITERCOUNT,stats.OUTERLOOPS);
end

if isfield(errorRecord,'contract_inside')
    fprintf(fid,'contract_inside\t\t%d:%f\n',errorRecord.time.contract_inside,errorRecord.contract_inside); 
    y = debugErrDump(errorRecord.contract_inside,errorRecord.data.contract_inside,debugLevel,dfile,'contract_inside',stats.ITERCOUNT,stats.OUTERLOOPS);
end

if isfield(errorRecord,'stochastic_min_max')
    fprintf(fid,'stochastic_min_max\t%d:%f\n',errorRecord.time.stochastic_min_max,errorRecord.stochastic_min_max);
    y = debugErrDump(errorRecord.stochastic_min_max,errorRecord.data.stochastic_min_max,debugLevel,dfile,'stochastic_min_max',stats.ITERCOUNT,stats.OUTERLOOPS);
end

if isfield(errorRecord,'stochastic_min_xbar')
    fprintf(fid,'stochastic_min_xbar\t%d:%f\n',errorRecord.time.stochastic_min_xbar,errorRecord.stochastic_min_xbar);
    y = debugErrDump(errorRecord.stochastic_min_xbar,errorRecord.data.stochastic_min_xbar,debugLevel,dfile,'stochastic_min_xbar',stats.ITERCOUNT,stats.OUTERLOOPS);
end

if isfield(errorRecord,'stochastic_max_xbar')
    fprintf(fid,'stochastic_max_xbar\t%d:%f\n',errorRecord.time.stochastic_max_xbar,errorRecord.stochastic_max_xbar);
    y = debugErrDump(errorRecord.stochastic_max_xbar,errorRecord.data.stochastic_max_xbar,debugLevel,dfile,'stochastic_max_xbar',stats.ITERCOUNT,stats.OUTERLOOPS);
end

if isfield(errorRecord,'shrink')
    fprintf(fid,'shrink\t%d:',errorRecord.time.shrink{1})
    for j=1:stats.fullN
        fprintf(fid,'%f\t',errorRecord.shrink{j});
        y = debugErrDump(errorRecord.shrink{j},errorRecord.data.shrink{j},debugLevel,dfile,['shrink_' num2str(j)],stats.ITERCOUNT,stats.OUTERLOOPS);
    end
    fprintf(fid,'\n');
end

if isfield(errorRecord,'mutate')
    fprintf(fid,'%s\t%d:',stats.how,errorRecord.time.mutate{1})
    for j=1:stats.fullN
        fprintf(fid,'%f\t',errorRecord.mutate{j});
        y = debugErrDump(errorRecord.mutate{j},errorRecord.data.mutate{j},debugLevel,dfile,['mutate_' num2str(j)],stats.ITERCOUNT,stats.OUTERLOOPS);
    end
end

fprintf(fid,'\n');
fclose(fid);

x = 1;

%--------------------------------------------------------------------------
function x = debugErrDump(specError,specErrorRecord,dlevel,file,name,iter,loop)
if dlevel > 1
    fid = fopen(file,'a');
    fprintf(fid,'%s\tITER:%d\tLOOP:%d\tERROR:%f\n',name,iter,loop,specError);
    for x = 1:size(specErrorRecord,1)
        for y = 1:size(specErrorRecord,2)
            fprintf(fid,'%64.64f\n',specErrorRecord(x,y));
        end
    end
    fprintf(fid,'\n============================================\n');
end

x = 1;

