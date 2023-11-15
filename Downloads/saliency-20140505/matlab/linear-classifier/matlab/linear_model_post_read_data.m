function [ldata,conf] = linear_model_post_read_data(ldata,conf)

dprint('Reading in raw data');

if(isfield(conf,'runNormalSet') && strcmp(conf.runNormalSet,'yes'))
    % Open Data into this
    ldata = get_ezvision_data(ldata,conf.imageDir,conf);
    ldata = get_att_gate_data(ldata,conf.imageDir,conf);
        
    %open subject data
    file = [conf.baseDir 'allPresent_S3.new.txt'];
    dprint(['Loading Subject Data - ' file]);   
    tprint('start');
    [ldata.S_NAME ldata.S_TFRAME ldata.S_SAMPLE ldata.S1 ldata.S2 ldata.S3 ldata.S4 ldata.S5 ldata.S6 ldata.S7 ldata.S8 ldata.S_VALUE] = ...
            textread(file,'%s %d %d %d %d %d %d %d %d %d %d %d');
    tprint('stop');
    conf.SUBJECT_COUNT         = 8;
    conf.hardBound             = 2;   % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 7;   % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    conf.midBound              = 5;
    conf.useCert               = 'no';
   
    % Sort the input subject data to align with the suprise input data
    dprint('Sorting Subject Data');   
    tprint('start');
    [ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.S_VALUE, ldata.TFRAME, ldata.SAMPLE);
    tprint('stop');   
    %ldata = get_m_class(ldata,conf);
end
    
% run all this again and cat hard data
if(isfield(conf,'runHardSet') && strcmp(conf.runHardSet,'yes'))    % Open Data into this
    file = [conf.hardDir 'chan.combined.',conf.condString,'.txt'];
    dprint(['Parsing Hard Data - ' file]); 
    tprint('start');
    [hdata.NAME hdata.TFRAME hdata.SAMPLE hdata.OLDFRAME hdata.NEWFRAME hdata.FEATURE ...
        hdata.MIN hdata.MAX hdata.AVG hdata.STD hdata.MAXX hdata.MAXY hdata.COND] =  ...
    	textread(file,'%s %d %d %d %d %s %f %f %f %f %d %d %d');
    tprint('stop');

    %open subject data
    file = [conf.baseDir 'allPresent_S3.hard.txt'];
    dprint(['Loading Subject Hard Data - ' file]);   
    tprint('start');
    [hdata.S_NAME hdata.S_TFRAME hdata.S_SAMPLE hdata.S1 hdata.S2 hdata.S3 hdata.S4 hdata.S5 hdata.S6 hdata.S7 hdata.S8 hdata.S_VALUE hdata.S_COND] = ...
        textread(file,'%s %d %d %d %d %d %d %d %d %d %d %d %d');
    tprint('stop');
    conf.SUBJECT_COUNT         = 8;
    conf.hardBound             = 2;    % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 7;    % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    conf.midBound              = 5;   
    conf.useCert               = 'no';
    
	% Sort the input subject data to align with the suprise input data
	dprint('Sorting Subject Data');   
	tprint('start');
	[hdata.CLASS hdata.CLASS_IDX] = linear_model_get_hard_class(hdata.S_TFRAME, hdata.S_SAMPLE, hdata.S_VALUE, hdata.S_COND, hdata.TFRAME, hdata.SAMPLE, hdata.COND);
	tprint('stop');
        
	%merge needed sets
	if(isfield(conf,'runHardSet') && strcmp(conf.runHardSet,'yes'))
        if(isfield(conf,'runNormalSet') && strcmp(conf.runNormalSet,'yes'))
            ldata.AVG      = [ldata.AVG; hdata.AVG];
            ldata.STD      = [ldata.STD; hdata.STD];
            ldata.MAXX     = [ldata.MAXX; hdata.MAXX];
            ldata.MAXY     = [ldata.MAXY; hdata.MAXY];
            ldata.NEWFRAME = [ldata.NEWFRAME; hdata.NEWFRAME];
            ldata.FEATURE  = [ldata.FEATURE; hdata.FEATURE];
            ldata.CLASS    = [ldata.CLASS; hdata.CLASS];
            ldata.TFRAME   = [ldata.TFRAME; hdata.TFRAME];
            ldata.SAMPLE   = [ldata.SAMPLE; hdata.SAMPLE];
            dprint('Hard and Normal Sets Merged');
        else
            ldata.AVG      = hdata.AVG;
            ldata.STD      = hdata.STD;
            ldata.MAXX     = hdata.MAXX;
            ldata.MAXY     = hdata.MAXY;
            ldata.NEWFRAME = hdata.NEWFRAME;
            ldata.FEATURE  = hdata.FEATURE;
            ldata.CLASS    = hdata.CLASS;
            ldata.TFRAME   = hdata.TFRAME;
            ldata.SAMPLE   = hdata.SAMPLE;
            dprint('Hard Sets Copied');
        end
	end
end

if(isfield(conf,'runMaskSet') && strcmp(conf.runMaskSet,'yes'))
	% Open Data into this    
    ldata = get_ezvision_data(ldata,conf.maskDir,conf);
    ldata = get_att_gate_data(ldata,conf.maskDir,conf);
    
	%open subject data
	file = [conf.baseDir 'allPresent_S3.new.txt'];
	dprint(['Loading Subject Data - ' file]);   
	tprint('start');
	[ldata.S_NAME ldata.S_TFRAME ldata.S_SAMPLE ldata.S1 ldata.S2 ldata.S3 ldata.S4 ldata.S5 ldata.S6 ldata.S7 ldata.S8 ldata.S_VALUE] = ...
        textread(file,'%s %d %d %d %d %d %d %d %d %d %d %d');
	tprint('stop');
    conf.SUBJECT_COUNT         = 8;
    conf.hardBound             = 1;  % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 8;  % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    conf.midBound              = 5;
    conf.useCert               = 'no';
            
	% Sort the input subject data to align with the suprise input data
	dprint('Sorting Subject Data');   
	tprint('start');
	[ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.S_VALUE, ldata.TFRAME, ldata.SAMPLE);
	tprint('stop');
    ldata = get_m_class(ldata,conf);
end

if(isfield(conf,'runTransMaskSet') && strcmp(conf.runTransMaskSet,'yes'))
	% Open Data into this    
    ldata = get_ezvision_data(ldata,conf.transMaskDir,conf);
    ldata = get_att_gate_data(ldata,conf.transMaskDir,conf);
    
    %open subject data
    file = [conf.transDir 'allPresent.txt'];
    dprint(['Loading Subject Data - ' file]);   
    tprint('start');
    [ldata.S_NAME ldata.S_TFRAME ldata.S_SAMPLE ldata.S1 ldata.S2 ldata.S3 ldata.S4 ldata.S_VALUE ldata.S_VALUE_NEG ...
     ldata.CERT_VALUE ldata.CERT_VALUE_POS ldata.CERT_VALUE_NEG] = ...
            textread(file,'%s %d %d %d %d %d %d %d %d %f %f %f');
    
    ldata.S_SAMPLE = ldata.S_SAMPLE + 1;
    ldata.SAMPLE   = ldata.SAMPLE   + 1;
    
    tprint('stop');
    conf.SUBJECT_COUNT         = 4; 
    conf.hardBound             = 1;  % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 4;  % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    conf.midBound              = 2;
    conf.useCert               = 'no'; 
    conf.certEasy              = 4.75; % Certanty Spotted is 4 to 6     
    conf.certHard              = 2.25; % Certanty Missed  is 3 to 1
                
	% Sort the input subject data to align with the suprise input data  
    dprint('Sorting Subject Data');   
    tprint('start');
    if(strcmp(conf.useCert,'yes'))
        [ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.CERT_VALUE, ldata.TFRAME, ldata.SAMPLE); 
    else
        [ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.S_VALUE, ldata.TFRAME, ldata.SAMPLE);
    end
    tprint('stop');  
    ldata = get_m_class(ldata,conf); 
end

if(isfield(conf,'runHardMaskSet') && strcmp(conf.runHardMaskSet,'yes'))
	% Open Data into this    
    ldata = get_ezvision_data(ldata,conf.hardMaskDir,conf);
    ldata = get_att_gate_data(ldata,conf.hardMaskDir,conf);
    
	%open subject data
	file = [conf.baseDir 'allPresent_S3.hard_mask.txt'];
	dprint(['Loading Subject Data - ' file]);   
	tprint('start');
	[ldata.S_NAME ldata.S_TFRAME ldata.S_SAMPLE ldata.S1 ldata.S2 ldata.S3 ldata.S4 ldata.S5 ldata.S6 ldata.S7 ldata.S8 ldata.S_VALUE ldata.S_DIFF] = ...
        textread(file,'%s %d %d %d %d %d %d %d %d %d %d %d %d');
	tprint('stop');
    conf.SUBJECT_COUNT         = 8;
    conf.hardBound             = 3;  % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 8;  % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    conf.midBound              = 5;
    conf.useCert               = 'no';
            
	% Sort the input subject data to align with the suprise input data
	dprint('Sorting Subject Data');   
	tprint('start');
	[ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.S_VALUE, ldata.TFRAME, ldata.SAMPLE);
	tprint('stop');
    ldata = get_m_class(ldata,conf);
end

if(isfield(conf,'runNotHardMaskSet') && strcmp(conf.runNotHardMaskSet,'yes'))
	% Open Data into this    
    ldata = get_ezvision_data(ldata,conf.notHardMaskDir,conf);
    ldata = get_att_gate_data(ldata,conf.notHardMaskDir,conf);
    
	%open subject data
	file = [conf.baseDir 'allPresent_S3.not-hard_mask.txt'];
	dprint(['Loading Subject Data - ' file]);   
	tprint('start');
	[ldata.S_NAME ldata.S_TFRAME ldata.S_SAMPLE ldata.S1 ldata.S2 ldata.S3 ldata.S4 ldata.S5 ldata.S6 ldata.S7 ldata.S8 ldata.S_VALUE ldata.S_DIFF] = ...
        textread(file,'%s %d %d %d %d %d %d %d %d %d %d %d %d');
	tprint('stop');
    conf.SUBJECT_COUNT         = 8;
    conf.hardBound             = 2;  % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 8;  % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    conf.midBound              = 5;
    conf.useCert               = 'no';
            
	% Sort the input subject data to align with the suprise input data
	dprint('Sorting Subject Data');   
	tprint('start');
	[ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.S_VALUE, ldata.TFRAME, ldata.SAMPLE);
	tprint('stop');
    ldata = get_m_class(ldata,conf);
end

if(isfield(conf,'runNewTransSet') && strcmp(conf.runNewTransSet,'yes'))
    % Open Data into this
    ldata = get_ezvision_data(ldata,conf.transDir,conf);
    ldata = get_att_gate_data(ldata,conf.transDir,conf);
    
    %open subject data
    file = [conf.transDir 'allPresent.txt'];
    dprint(['Loading Subject Data - ' file]);   
    tprint('start');
    [ldata.S_NAME ldata.S_TFRAME ldata.S_SAMPLE ldata.S1 ldata.S2 ldata.S3 ldata.S4 ldata.S_VALUE ldata.S_VALUE_NEG ...
     ldata.CERT_VALUE ldata.CERT_VALUE_POS ldata.CERT_VALUE_NEG] = ...
            textread(file,'%s %d %d %d %d %d %d %d %d %f %f %f');
    
    ldata.S_SAMPLE = ldata.S_SAMPLE + 1;
    ldata.SAMPLE   = ldata.SAMPLE   + 1;
    
    tprint('stop');
    conf.SUBJECT_COUNT         = 4; 
    conf.hardBound             = 1;  % Where do we call sequences hard (0-8) would be 2 JoV / JOSA?
    conf.easyBound             = 4;  % Where do we call seqeunces easy (0-8) would be 7 JoV / JOSA?
    conf.midBound              = 2;
    conf.useCert               = 'no'; 
    conf.certEasy              = 4.75; % Certanty Spotted is 4 to 6     
    conf.certHard              = 2.25; % Certanty Missed  is 3 to 1
    
    % Invert the metric so that high values are Easy and low values are Hard
    ldata.CERT_VALUE           = abs(ldata.CERT_VALUE - 6);
    
    % Sort the input subject data to align with the suprise input data
    dprint('Sorting Subject Data');   
    tprint('start');
    if(strcmp(conf.useCert,'yes'))
        [ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.CERT_VALUE, ldata.TFRAME, ldata.SAMPLE); 
    else
        [ldata.CLASS ldata.CLASS_IDX] = linear_model_get_class(ldata.S_TFRAME, ldata.S_SAMPLE, ldata.S_VALUE, ldata.TFRAME, ldata.SAMPLE);
    end
    tprint('stop');  
    ldata = get_m_class(ldata,conf); 
end

%--------------------------------------------------------------------------

function ldata = get_ezvision_data(ldata,dir,conf)

file = [dir 'chan.combined.',conf.condString,'.txt'];
dprint(['Parsing Data - ' file]); 
tprint('start');
[ldata.NAME ldata.TFRAME ldata.SAMPLE ldata.OLDFRAME ldata.NEWFRAME ldata.FEATURE ...
        ldata.MIN ldata.MAX ldata.AVG ldata.STD ldata.MAXX ldata.MAXY ...
        ] =  ...
        textread(file,'%s %d %d %d %d %s %f %f %f %f %d %d');
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

%--------------------------------------------------------------------------
% sub sample the class list
function ldata = get_m_class(ldata,conf)
dprint('Getting mask class sub sample');
for i=1:size(ldata.M_SAMPLE,1)
    ldata.M_CLASS(i,1) = ldata.CLASS(i,1);  
end
% get Easy / Hard classification
ldata.M_NEW_CLASS = linear_model_get_new_class(ldata.M_CLASS,conf);

