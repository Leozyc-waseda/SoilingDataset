function conf = linear_model_post_process_perl(conf)

if(isfield(conf,'runNormalSet') && strcmp(conf.runNormalSet,'yes')) 
    imageDir = conf.imageDir;
elseif(isfield(conf,'runNewTransSet') && strcmp(conf.runNewTransSet,'yes'))
    imageDir = conf.transDir;
elseif(isfield(conf,'runHardSet') && strcmp(conf.runHardSet,'yes'))
    imageDir = conf.hardDir;
elseif(isfield(conf,'runMaskSet') && strcmp(conf.runMaskSet,'yes'))
    imageDir = conf.maskDir;
 elseif(isfield(conf,'runTransMaskSet') && strcmp(conf.runTransMaskSet,'yes'))
    imageDir = conf.transMaskDir;   
elseif(isfield(conf,'runHardMaskSet') && strcmp(conf.runHardMaskSet,'yes'))
    imageDir = conf.hardMaskDir;
elseif(isfield(conf,'runNotHardMaskSet') && strcmp(conf.runNotHardMaskSet,'yes'))
    imageDir = conf.notHardMaskDir;
else
    error('Cannot determine set type for running perl post process');
end


if isfield(conf,'extraPerlArgs')
    command = ['perl ' conf.baseDir 'script/step_1_parse_new_files.pl ' conf.extraPerlArgs ' --out=' conf.condString ' --base=' imageDir];
elseif strcmp(conf.condString,'UCIO_legacy') || strcmp(conf.condString,'UHIO_legacy') || ...
       strcmp(conf.condString,'UHIOLTX_legacy') || strcmp(conf.condString,'UHIOGKSE_legacy')    
        command = ['perl ' conf.baseDir 'script/step_1_parse_new_files.pl -o --out=' conf.condString ' --base=' imageDir];
else
    if isfield(conf,'getFrequencyData') && strcmp(conf.getFrequencyData,'yes')
        command = ['perl ' conf.baseDir 'script/step_1_parse_new_files.pl -f --out=' conf.condString ' --base=' imageDir];
    else
        command = ['perl ' conf.baseDir 'script/step_1_parse_new_files.pl ' conf.condString ' ' imageDir];
    end
end
    
dprint(['Running Perl Script - ' command]);
tprint('start');
[pl_status,pl_result] = unix(command);
tprint('stop'); 
    
% parse the attention gate / ground truth mask files stuff
if((isfield(conf,'runMaskSet')        && strcmp(conf.runMaskSet,'yes'))        || ...
   (isfield(conf,'runHardMaskSet')    && strcmp(conf.runHardMaskSet,'yes'))    || ...
   (isfield(conf,'runNotHardMaskSet') && strcmp(conf.runNotHardMaskSet,'yes')) || ...
   (isfield(conf,'runTransMaskSet')   && strcmp(conf.runTransMaskSet,'yes')))
    command = ['perl ' conf.baseDir 'script/step_2_parse_mask.pl --out=' conf.condString ' --base=' imageDir ' --use_full_mask'];
else
    command = ['perl ' conf.baseDir 'script/step_2_parse_mask.pl --out=' conf.condString ' --base=' imageDir];
end
     
dprint(['Running Perl Script - ' command]);
tprint('start');
[plm_status,plm_result] = unix(command);
tprint('stop');

