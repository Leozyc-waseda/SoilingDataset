function conf = linear_model_post_process_perl_AB(conf)




if isfield(conf,'extraPerlArgs')
    command = ['perl ' conf.baseDir 'script/step_1_parse_AB_files.pl ' conf.extraPerlArgs ' --out=' conf.condString ' --base=' conf.ABDir];
else
    command = ['perl ' conf.baseDir 'script/step_1_parse_AB_files.pl --out=' conf.condString ' --base=' conf.ABDir];
end
    
dprint(['Running Perl Script - ' command]);
tprint('start');
[pl_status,pl_result] = unix(command);
tprint('stop'); 
    
% parse the attention gate / ground truth mask files stuff
%%% BLANK FOR NOW
     


