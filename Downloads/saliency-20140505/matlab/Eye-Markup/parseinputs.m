function settings = parseinputs(varargin)
% 
% PARSEINPUTS  accepts any series of key/value parameters and
% returns settings for markup of .eye files
%
% PARSEINPUTS returns a default structure of parameters for
% markup. The default set of arguments is described and used
% in (Dave's citation)  Read the source for descriptions of each.
% 
% PARSEINPUTS(ALIAS,...) returns a structure of custom-set
% parameters.  Read the source file for the list of adjusted arguments.
%
% PARSEINPUTS(PARAMETER,VALUE,...) returns a structure with
% individual parameters set.  This is equivalent to getting the
% default parameter list and then setting params.PARAMETER = VALUE.
% VALUE can be a number, string or array as needed by the field.
% ALIASES can also be used with this command.
% 
% If multiple instances of a parameter exist in the command line due to
% double typing or aliases then the last instance will be chosen as the
% setting, with individual arguments taking priority over aliases.  
% To overide part of an alias put your command after it.
%
% PARSEINPUTS(SETTINGS,PARAMETER,VALUE,...), where SETTINGS is a
% previous parameter structure, returns an updated parameter structure
% with new values assigned as above. 
% 
% Use getvalue(<'parameter'>,parseinputs); to get a specific
% parameter's default.  It is best not to edit the parameter
% structure directly.  
%
% See also GETVALUE.

% return default parameters
if(nargin == 0)
  settings = defaultparams;
  return;
end

% detect if last argument is already a setting struct
if(isfield(varargin{1},'stats'))
  settings = varargin{1};
  if (nargin == 1)
    return;
  end
  arg_queue = varargin(2:end);
else
  settings = defaultparams;
  arg_queue = varargin;
end


alias_presets = aliasList;
alias_list = alias_presets(:,1);

%%% sort between aliases and individual assignments

%%% TODO: fix multiple identical alias bug
% for comparison purposes, find strings in the queue
i_strargs = find(cellfun(@ischar, arg_queue));
% separate the aliases from the regular guys
[aliases,i_alias] = intersect(arg_queue(i_strargs), alias_list);
arg_indiv = arg_queue(setdiff(1:end,i_strargs(i_alias)));

% convert aliases to options
arg_preset = [];
for ii = 1:length(aliases);
  k = strmatch(aliases{ii},alias_list);
  arg_preset = [arg_preset alias_presets{k,2}];
end

%%% assign the options their values

% give individual assignments
var_lists = [arg_preset, arg_indiv];
for i = 1:2:length(var_lists)
  if (isfield(settings, var_lists{i})) % we can only write to
                                       % existing options
    settings.(var_lists{i})=var_lists{i+1};
  else
    error('parseinputs:badOption','Unknown parameter %s',var_lists{i});
  end
end

%%% manual fixes

%fix ppd if user only entered one value
if (length(settings.ppd) == 1) 
  f = settings.ppd;
  settings.ppd = [f f];
end  

