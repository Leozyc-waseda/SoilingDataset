function [eyedata,newparams] = importEyeData(fname,params)
% This function imports eyedata into a proper structure 
% and also reads header parameters that may be commented as ##
% It replaces the loadCalibTxt function

% The input file should be .eye or .eyeS, text delimited fields

% fields of the returned eyedata structure are
%  N
%  xy = 2xN eye position  
%  pd = 1xN integer
%  vel = 1XN 
%  status = 1xN integer (initialized as not classified)
% newparams is a cell array which should be passed to parseinputs
%
% Also, custom arguments for each file can be placed in the header
% as # <argname> argvalue
% is equivalent to 'params.argname=argvalue'
% argvalue can be an array, no quotes necessary for strings
% 
%written: Nov 2009
%author : John Shen
%iLab - University of Southern California
%**************************************************************************

if(nargin == 1) params = defaultparams; end

%%% TODO: import loadCalibTxt-type setup-contingent input
%%% currently nips-monkey input is not supported, 
%%% i.e. we assume that the data is of format x y pdiam for each row
puprange = getvalue('pup_range',params);
vblevel = getvalue('verbose',params);
V = getvalue('verboselevels',params);
scode = getvalue('code',params);

%%% open file
[ferp message] = fopen(fname, 'r');
if(~isempty(message))
    fclose(ferp);
    error('importEyeData:FileNotOpened', ... 
	'File ''%s'' can not be opened because: %s', fname, message);
end

% test if file is empty or not
if all(fgetl(ferp) == -1)
    error('importEyeData:EmptyFile','File %s is empty.', fname);
    eyedata=[];
    newparams=params;
    return;
else
    fseek(ferp,0,-1);
end
%%% read file headers with format as 

% # argname argvalue 
header = textscan(ferp, '# %s %80[^\n]');
arg_names = header{1}; % takes in first column
arg_vals = header{2};

% convert numbers/arrays but not strings to strings
[arg_vals is_num] = cellfun(@str2num,arg_vals,'UniformOutput', ...
			    false);
% copy strings to the value array
isnot_num = ~[is_num{:}]; % cell array fun
arg_vals(isnot_num) = header{2}(isnot_num);

% construct cell array of names/vals
n_args = length(arg_names);
newargs = cell(1,n_args*2);
newargs(1:2:2*n_args-1) = arg_names; 
newargs(2:2:2*n_args  ) = arg_vals;

if vblevel>=V.SUB && ~isempty(arg_names)
  for i = 1:numel(arg_names)
        fprintf('\tCustom parameter from %s: \n\t\t%s = %s\n', fname, char(arg_names(i)),num2str(arg_vals{i}));
    end
end

% get new setting structure
newparams = parseinputs(params,newargs{:});

%%% import the rest of data
rawdata = textscan(ferp, ''); 
% reads data by space-delimed columns 
% into a 1xC cell array of row matrices
fclose(ferp);                  

n_fields = length(rawdata); 
if n_fields < 2
  error('importEyeData:BadEyeFile',...
	'Not enough fields in file %s or improper comment', fname);  
end

%%% populate data
eyedata.len = length(rawdata{1});
if eyedata.len == 0
  error('importEyeData:FileNotRead',...
	'File %s is not readable', fname);  
end

% fill data structure with row matrices
eyedata.xy = [rawdata{1:2}]'; % in screen coordinates

% at this point we can also read in data we want to import

% do we have the pupil diameter field?
if n_fields >= 3
  eyedata.pd = rawdata{3}'; % put it in
else
  % put filler in for the pupil diameter 
  % it's not used for our analysis anyway, and if it is you should be 
  % putting it in as data!
  eyedata.pd = sum(puprange)/2*ones(1,eyedata.len);
  
  % optional code: take out the diameter from reported stats
  % stats = getvalue('stats',params);
  % parseinputs(params, 'stats', {stats{~strcmp(stats,'pd')}}); 
  if vblevel >= V.TOP
      fprintf([ '\tNote: No pupil diameter data was found..\n' ...
                '\tWarning: filler pupil diameter data added.\n']);
  end
end
% don't handle the other fields

% pre-calculate instantaneous velocity in degrees/sec
eyedata.vel = getVel(eyedata,newparams);
% set status to fixation for now
eyedata.status = scode.NC*ones(1,eyedata.len);

% report
if vblevel>=V.SUB
    fprintf('\t%d samples loaded into data\n', eyedata.len);
end