function [ldata,conf] = linear_model_post_read_data_AB(ldata,conf)

dprint('Reading in raw data');

% Open Data into this
ldata = get_ezvision_data(ldata,conf.ABDir,conf);
%ldata = get_att_gate_data(ldata,conf.imageDir,conf);
        
%open subject data
file = [conf.baseDir 'subject_resp.txt'];
dprint(['Loading Subject Data - ' file]);   
tprint('start');
[ ldata.S_SAMPLE_ABS ldata.S_SAMPLE ldata.S_TARGETS ldata.S_CONDITION ldata.S_TFRAME_A ldata.S_TFRAME_B ldata.S_TFRAME_DIFF         ...
  ldata.S_A_CB ldata.S_B_CB ldata.S_A_JM ldata.S_B_JM ldata.S_A_KK ldata.S_B_KK ldata.S_A_KS ldata.S_B_KS ldata.S_A_LT ldata.S_B_LT ...
  ldata.S_VALUE_A ldata.S_VALUE_B ldata.S_AVG_POS_A ldata.S_AVG_POS_B ] = ...
            textread(file,'%d%d%d%s%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%f%f','delimiter',',');
tprint('stop');

ldata.S_SAMPLE = ldata.S_SAMPLE + 1;
ldata.A.SAMPLE = ldata.A.SAMPLE + 1;
ldata.B.SAMPLE = ldata.B.SAMPLE + 1;

conf.SUBJECT_COUNT         = 5;
conf.midBound              = 3;
conf.useCert               = 'no';
   
% Sort the input subject data to align with the suprise input data
% Also get the offset difference between target A and B
if strcmp(conf.useCert,'no')
    dprint('Sorting Subject Data by Base Detection');   
    tprint('start');
    conf.hardBound             = 1;   % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 5;   % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    [ldata.A.CLASS ldata.A.CLASS_IDX ldata.A.TFRAMED ldata.A.TFRAMED_IDX ldata.A.S_ABType] = ...
        linear_model_get_class_AB(ldata.S_TFRAME_A, ldata.S_TFRAME_B, ldata.S_VALUE_A, ldata.S_TFRAME_DIFF, ...
        ldata.S_CONDITION, ldata.A.ABType, ldata.A.TFRAME_A, ldata.A.TFRAME_B, ldata.S_SAMPLE, ldata.A.SAMPLE);
    [ldata.B.CLASS ldata.B.CLASS_IDX ldata.B.TFRAMED ldata.B.TFRAMED_IDX ldata.B.S_ABType] = ...
        linear_model_get_class_AB(ldata.S_TFRAME_B, ldata.S_TFRAME_A, ldata.S_VALUE_B, ldata.S_TFRAME_DIFF, ...
        ldata.S_CONDITION, ldata.B.ABType, ldata.B.TFRAME_B, ldata.A.TFRAME_A, ldata.S_SAMPLE, ldata.B.SAMPLE);
    tprint('stop');   
else
    dprint('Sorting Subject Data by Detection with Certanty'); 
    tprint('start');    
    conf.hardBound             = 1;   % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 4;   % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    [ldata.A.CLASS ldata.A.CLASS_IDX ldata.A.TFRAMED ldata.A.TFRAMED_IDX ldata.A.S_ABType] = ...
        linear_model_get_class_AB(ldata.S_TFRAME_A, ldata.S_TFRAME_B, abs(ldata.S_AVG_POS_A - 6), ldata.S_TFRAME_DIFF, ...
        ldata.S_CONDITION, ldata.A.ABType, ldata.A.TFRAME_A, ldata.A.TFRAME_B, ldata.S_SAMPLE, ldata.A.SAMPLE);
    [ldata.B.CLASS ldata.B.CLASS_IDX ldata.B.TFRAMED ldata.B.TFRAMED_IDX ldata.B.S_ABType] = ...
        linear_model_get_class_AB(ldata.S_TFRAME_B, ldata.S_TFRAME_A, abs(ldata.S_AVG_POS_B - 6), ldata.S_TFRAME_DIFF, ...
        ldata.S_CONDITION, ldata.B.ABType, ldata.B.TFRAME_B, ldata.A.TFRAME_A, ldata.S_SAMPLE, ldata.B.SAMPLE);
    tprint('stop');  
end

%--------------------------------------------------------------------------

function ldata = get_ezvision_data(ldata,dir,conf)

file = [dir 'A.chan.combined.',conf.condString,'.txt'];
dprint(['Parsing Data - ' file]); 
tprint('start');
[ldata.A.NAME ldata.A.TFRAME_A ldata.A.TFRAME_B ldata.A.TFRAME_DIFF ldata.A.SAMPLE ldata.A.VAR ldata.A.ABType ...
        ldata.A.OLDFRAME ldata.A.NEWFRAME ldata.A.TTYPE ldata.A.FEATURE ...
        ldata.A.MIN ldata.A.MAX ldata.A.AVG ldata.A.STD ldata.A.MAXX ldata.A.MAXY ldata.A.BLANK...
        ] = ...
        textread(file,'%s %d %d %d %d %d %d %d %d %d %s %f %f %f %f %d %d %d');
dprint(['Lines - ' num2str(size(ldata.A.NAME,1))]);
tprint('stop');

file = [dir 'B.chan.combined.',conf.condString,'.txt'];
dprint(['Parsing Data - ' file]); 
tprint('start');
[ldata.B.NAME ldata.B.TFRAME_A ldata.B.TFRAME_B ldata.B.TFRAME_DIFF ldata.B.SAMPLE ldata.B.VAR ldata.B.ABType ...
        ldata.B.OLDFRAME ldata.B.NEWFRAME ldata.B.TTYPE ldata.B.FEATURE ...
        ldata.B.MIN ldata.B.MAX ldata.B.AVG ldata.B.STD ldata.B.MAXX ldata.B.MAXY ldata.B.BLANK...
        ] = ....
        textread(file,'%s %d %d %d %d %d %d %d %d %d %s %f %f %f %f %d %d %d');
dprint(['Lines - ' num2str(size(ldata.B.NAME,1))]);
tprint('stop');

%--------------------------------------------------------------------------

function ldata = get_att_gate_data(ldata,dir,conf)

file = [dir 'mask.',conf.condString,'.txt'];
dprint(['Parsing Data - ' file]); 
tprint('start');

[ldata.M_NAME ldata.M_TFRAME ldata.M_SAMPLE ldata.M_OLDFRAME ldata.M_NEWFRAME ...
        ldata.M_TOTALCOUNT ldata.M_MASKCOUNT ldata.M_LAMCOUNT ldata.M_OVERLAPCOUNT ...
        ldata.LAM_MIN  ldata.LAM_MAX  ldata.LAM_AVG  ldata.LAM_STD ...
        ldata.LAM_MINX ldata.LAM_MINY ldata.LAM_MAXX ldata.LAM_MAXY ldata.LAM_N ...
        ] =  ...
        textread(file,'%s %d %d %d %d %d %d %d %d %f %f %f %f %d %d %d %d %d');
tprint('stop'); 


