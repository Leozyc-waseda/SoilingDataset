function result=markEye(glob,varargin)
%MARKEYE eye-markup utility
%   MARKEYE(GLOB, PIPE, VARARGIN) marks or re-marks an eye or eyeS file. 
%   Marks eye movement areas as:
%
%       fixation:                   0 blue
%       saccade:                    1 green
%       blink/Artifact:             2 red
%       Saccade during Blink:       3 green (dot-dash line style)
%       smooth pursuit:             4 magenta
%       drift/misclassification:    5 black
%       combined saccades:          6 cyan
%       invalid                     NaN dotted
%
%   This is the main calling function for this package of utilities.  It takes
%   as its first argument the directory to an eye or eyeS file[s] and creates a
%   ceyeS file[s] containing the markup.  This file is ready for ezvision
%   or superimposer. Additional arguments are specified in a string/value
%   order.  see example below.
%
%   Inputs:    
%       GLOB: opens all files matching the pattern GLOB for reading.
%       Supports standard globbing in Unix/Mac (?), single wildcards in PC.
%
%       PIPENAME: calls different 'pipelines' according to the type 
%       of markup needed.  Defaults to revised version of Dave's JoV 2009
%       analysis if not specified.
%      
%   Optional inputs:
%       See 'parseinputs.m' for all command line options.
%
%   Examples:
%       This marks all .eye files in the path, with a sampling frequency of 1kHz.
%           markeye('<path>/*.eye','sf',1000);
%         
%       This should plot the eyetrace of an e-ceyeS file.  
%           markEye('*.eyeS','plot_ceyeS') 
%       Note that pipelines now allow different markEye 'routines' to be run, 
%       greatly increasing versatility.  The called pipeline must be the first 
%       arguments after the eye glob.  If no pipeline, we simply run dave's
%       JoV 2009 analysis.
%
%   See also PARSEINPUTS, DEFAULTPARAMS, RUNPIPELINE, EXPORTEYEDATA.
%   written: Sept 2006
%   author : David J. Berg
%   modified: John Shen (Oct 2009, Apr 2010)

%improvements:
% (1) structures instead of paired cell arrays store data and parameters
% (2) program runs as a 'pipeline' such that
% different subroutines can be called up according to the
% information/processing desired
% examples of code below
% (3) adding 'debug','show_trace','quiet','silent' gives different levels
% of information
% (4) named pipelines: see above.  effectively allows the same control over
% runtime flow as aliases did for options.
% (5) multi-core threading: 'dual_core' and 'quad_core' aliases and
% 'ncores' options set up parallel cores in matlab, only tested on R2010a
% up multi-core 
%iLab - University of Southern California
%**************************************************************************

% get path access to utilities
setpaths;

% TODO: make automagically generated help file for documentation
% and improve documentation of m-files for public consumption

% get a list of the files
if (nargin < 1)
    error('markEye:noFile','At least a path/filename is needed.');
end

fprintf('Reading files %s...\n',glob)
filelist = alldir(glob);
fprintf('%d files found.\n**********************\n',numel(filelist));

% take care of base-level parameters
routine_name = 'human_std'; %default
prompt_args = varargin;
if nargin > 1 && any(strcmp(varargin{1},pipeLib('names')))
    routine_name = varargin{1};
    prompt_args(1) = [];
end

% get the pipeline
[pipeline pipe_args] = pipeLib(routine_name); 
args = [pipe_args prompt_args]; % prompt can override presets

% set the parameters
base_params = parseinputs(args{:});
out_ext = getvalue('out_ext',base_params);
parcores = getvalue('ncores',base_params);

data = [];
if parcores == 1 %TODO: do some error checking here to make sure we
		 %don't ask for too many		
    for ii = 1:length(filelist);
      % get input and output file name
      % current output file destination is to the local directory of the input file  
      inputfile = filelist{ii};
      [f_pathstr f_stem] = fileparts(inputfile);
      outputfile = fullfile(f_pathstr, [f_stem '.' out_ext]);

      fprintf('Loading [%s] (%d/%d)...\n', inputfile,ii,length(filelist));
      [data,params] = submain(inputfile, outputfile, pipeline, base_params);
    end % end file loop

else
    matlabpool('local',parcores); % prep the cores
    parfor ii = 1:length(filelist);
        % get input and output file name
        % current output file destination is to the local directory of the input file  
        inputfile = filelist{ii};
        [f_pathstr f_stem] = fileparts(inputfile);
        outputfile = fullfile(f_pathstr, [f_stem '.' out_ext]);
    
        fprintf('Loading [%s] (%d/%d)...\n', inputfile,ii,length(filelist));
        [data,params] = submain(inputfile, outputfile, pipeline, ...
			    base_params);
    end % end file loop
    matlabpool('close');
end
% loop over our files


% output data to matlab
if length(filelist) == 1
    result = data;
else
    result = 'Done!'; % placeholder, 
    % TODO: accumulate events of interest, 
    % possibly w/ preallocation
end
end

function [data,params] = submain(inputfile, outputfile, pipeline, params)
no_overwrite = getvalue('skipfile',params);
overwrite = getvalue('overwrite',params);
is_exporting = getvalue('exportfile',params);
vblevel = getvalue('verbose',params);
V = getvalue('verboselevels',params);

doit = true;
% if the file already exists
if ~isempty(dir(outputfile)) && is_exporting
  doit = false;
  fprintf('%s is already written: ',outputfile);
  if overwrite 
    fprintf('overwriting... \n');
    doit = true;
  elseif no_overwrite
    fprintf('skipping file... \n');
    doit = false;
  else % ask
    s = input('overwrite? (y/n) [RET=no];','s');
    doit = ~(isempty(s) || lower(s(1)) ~= 'y') % do not ovewrite
  end
end

if doit
  % do it!
  [data, params] = runPipeline(pipeline,inputfile,params);
    
  % export the markup!
  if getvalue('exportfile',params)
    exportEyeData(data, params, outputfile);
    if vblevel>=V.USER
      disp('Press any key to continue...');
      pause
    end
  end
else
  data = [];
end

fclose('all'); % just in case we forget to close any files
end
