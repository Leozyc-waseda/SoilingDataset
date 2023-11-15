function settings = defaultparams
% Returns default parameters for eye-markup.  Edit with caution.

%**************************************************************
%general parameters
%**************************************************************
settings.sf=240;                  % sampling frequency (Hz)
settings.ppd=[35.1 33.075];       % [x y] pixels per degree of visual angle
settings.screen_size=[1920 1080]; % (pixels) resolution of the screen
settings.autosave=true;           % 1 to save files without prompting
settings.out_ext='e-ceyeS';       % the extension of the output file 
settings.skipfile=true;           % never overwrite an output file 
settings.overwrite=false;         % always overwrite an output file 
                                  % (warning: takes precedence)
settings.exportfile=true;         % should we export a file at all?
settings.trash=0;                 % how many samples to trash -
                                  % should be set here
settings.ncores=1;                % how many cores to run (1 means
                                  % no parallel running)
settings.alignstimulitime=0;      % if true, the timming of timeon and 
                                  % timeoff is aligned to the timing of
                                  % stimuli, rather than the time from the
                                  % beginning of the .eye file
                                  
%**************************************************************
% coding/reporting parameters
%**************************************************************
%stats to output in marked file
settings.stats={'x' 'y' 'pd' 'status' 'targetx', 'targety','ampl', ...
                'peakvel', 'timeon' 'timeoff' 'interval' 'typenum' }; 

% status codes for various eye movement events
settings.code = struct(...
    'FIXATION', 0,...
    'SACCADE',  1,...
    'BLINK',    2,...
    'SACBLINK', 3,...
    'SMOOTH',   4,...
    'NC',       5,...
    'SAC_CMBND',6);           

% do not change this code:
settings.code_nums = cellfun(@(x)settings.code.(x),fieldnames(settings.code))';

%**************************************************************
% export parameters
%**************************************************************
settings.event_stat_marker = '*';
settings.export_conf_file = 'export.mconf'; %TODO: deprecate
settings.events_to_stat = settings.code.SACCADE;%event to report
                                                %stats for
settings.write_col_header = true; % write column header

%**************************************************************
% plotting parameters
%**************************************************************
settings.code_color = {'b','g','r','y','m','k','c'}; % colors
settings.plot_screenwin = [ 0 1920 0 1080]; % window size plot margins

%**************************************************************
%eyetrace cleaning parameters
%**************************************************************
settings.lot_width=3;           %half-width of the window in samples
                                %to excise upon loss of tracking-event

settings.pup_range=[50 1000];   %minimum and maximum pupil size (in pixels)

settings.pup_maxstep=100;       %maximum pupil increase per sample

settings.max_vel=1000;          %maximum travel of the POR (in deg/s)
                                %maximum saccade velocity is 500 deg/s

settings.smooth_window = 7;     %full-width of smoothing binomial window

%**************************************************************
%saccade detection related parameters
%**************************************************************
settings.blink_window=60;       %size of the window to look for blinks
                                %(ms)

settings.sac_minvel=30;         %minimum velocity of saccade in
                                %degrees per second

settings.sac_minamp=2;          %min amplitude of saccade length in
                                %degrees

settings.sac_filter=63;         %lowpass cuttoff of filter to smooth
                                %eye movements.

settings.sac_mintime=20;        %saccades less than 'sac_mintime' (ms)
                                %in length are removed

settings.sac_pcathresh=.2;      %threshold for pca filter (higher
                                %value more saccades)

settings.sac_pcawindow=[10 20]; %pca window sizes used to detect
                                %saccades.

settings.maxsaclength=400;      %saccades longer then this threshold
                                %arebogus

%**************************************************************
%saccade combination related parameters
%**************************************************************
settings.pro_timethresh=60;    %fixations < threshold in time are
                               %considered for elimination during
                               %saccade combination i.e. minimum
                               %fixation duration for elimination (ms)

settings.pro_timewindow=15;    %The window in ms to look back/forward
                               %into the before/after fixation when
                               %determining if a fixation is part of a
                               %prosaccade (ms)

settings.pro_anglethresh=45;   %if the angle between the before and
                               %after saccade trajectories is less then
                               %pro_anglethresh it is accepted as a
                               %single saccade (deg) i.e max angle
                               %deviation

settings.pro_linearthresh=1.5; %if the fit residuals of the sampled
                               %saccade points is greater than
                               %pro_linearthresh in pixels then the
                               %saccade is rejected as a candidate for
                               %combination

settings.pro_singlepoint=false;%if set to 0, do not try to fit a line
                               %if there are too few points

%**************************************************************
%saccade cleanup and event combination parameters
%**************************************************************
settings.clean_window=80;    %window to use when combining events (ms)
settings.clean_keepfix=true; %keep fixations, or attempt to clean (1 for yes)
settings.event_window=5;     %size of event window (ms)

%**************************************************************
%smooth pursuit, fixation parameters
%**************************************************************
settings.smf_thresh=0.021;     %threshold to determine smooth pursuit from
                               %fixation

settings.smf_prefilter=63;     %filter to apply before analysis

settings.smf_window=[50 100];  %window sizes for PCA filter

%**************************************************************
%usage parameters
%**************************************************************
%give different levels of output
settings.verboselevels = struct(...
    'SILENT'    , -Inf,...   % shows nothing
    'BATCH'     , -1,...     % shows only which file is being run (batch)
    'TOP'       , 0,...      % shows the top-level commands being run
    'PLOT'      , 1,...      % plots the eye trace 
    'SUB'       , 2,...      % shows output of subroutines
    'USER'      , 3,...      % allows user to stop at key points
    'DATA'      , 5);        % allows user to see data... 
                             % Inf shows everything

settings.verbose=settings.verboselevels.TOP;

%**************************************************************
% misc parameters                            
%**************************************************************
settings.offscreenmargin=0;      %margin of screen to mark as off
                                 %screen(in degrees)

settings.modelfree_strict=false; %When we perform model free
                                 %statistics generally we ignore
                                 %saccades in blink and combined
                                 %saccades. Saccades in blink are
                                 %sometimes errors in the detection of
                                 %the cornial reflection which causes
                                 %the eye position to change at high
                                 %velocity. This option changes how
                                 %statistics of intersaccadic
                                 %intervals are treated when a pure
                                 %saccade has a saccade in blink or
                                 %combined saccade in its
                                 %intersaccadic interval. If this
                                 %option is set to 1 then we report 0
                                 %for the duration of the
                                 %intersaccadic interval if a event
                                 %type 3 or 6 is in it. If set to 0 we
                                 %just ignore the event type 3 or 6
                                 %and report the intersaccadic
                                 %interval as the duration till the
                                 %pure next saccade. [0]
settings.give_col_header=1; % output a column header
