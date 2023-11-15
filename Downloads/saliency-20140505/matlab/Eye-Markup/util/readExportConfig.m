function exSettings = readExportConfig(fname)
% This function reads out the export configurations
% It returns a structure array with the configuration data
% that exportEyeData reads.
%
% Formatting of the .conf file is in columns as follows 
% % comments are ignored
% {key aliases} is_on_event(bool) 'reference' format(%.1f)
%
% Defaults kick in if there are fewer fields.
%
% 
% written: Nov 2009
% author : John Shen
% iLab - University of Southern California
%*****************************************************************
% TODO: This needs to be replaced with a cell/struct array like the piple
% library
if(nargin < 1)
  error('readExportConfig:NoFile', 'No file name provided.');
end

%%% open file
[ferp message] = fopen(fname, 'r');
if(~isempty(message))
  error('readExportConfig:FileNotOpened', ... 
	'File ''%s'' can not be opened because: %s', fname, message);
end

% header is fieldnames but those are 
% more for the user than for the program
foo = textscan(ferp, '%% %100[^\n]');

regexes = {'{(?<aliases>[\w ]+)}',  ...
	   '(?<is_on_event>\d)', ...
	   '''(?<reference>[\w@ .,:_(){}]*)''', ...
	   '(?<format>%[\w.]+)'};
    
% read one line at a time because parsing can be messy
ii = 1;
while ~feof(ferp)
  line = fgetl(ferp);
  if isempty(line), continue; end; % skip empty lines
  
  tokens = struct([]);
  
  numtoks = length(regexes);
  while (isempty(tokens) && numtoks > 0)
    regline = strjoin(regexes(1:numtoks));
    tokens = regexp(line, regline, 'names');
    numtoks = numtoks-1;
  end

  if numtoks == 0 && isempty(tokens)
    error('readExportConfig:badFile', 'unreadable line %s', line);
  end
  
  % reformat a few things
  tokens.aliases = textscan(tokens.aliases, '%s');
  tokens.aliases = tokens.aliases{:};

  % set key field
  tokens.key = tokens.aliases{1};

  % defaults for is_on_event 
  if ~isfield(tokens, 'is_on_event')
    tokens.is_on_event = 0;
  else
    tokens.is_on_event = str2num(tokens.is_on_event); 
  end


  % parse referent - this shows how to access the data
  % should be one liner - see export.mconf
  if ~isfield(tokens,'reference')
    tokens.reference = tokens.key;
  elseif isempty(tokens.reference)
    tokens.reference = tokens.key;
  elseif tokens.reference(1) == '@'
    if(matlabVersion >= 7.8) %R2009a
      tokens.reference = str2func(tokens.reference);
    else
      % sad workaround for anon funcs
      eval(sprintf('foo = %s;', tokens.reference)); 
      tokens.reference = foo;
    end
  end


  % defaults for format - should be parameter 
  if ~isfield(tokens, 'format')
    tokens.format = '%.1f';
  end
  tokens = orderfields(tokens);

  code(ii) = tokens;
  ii = ii+1;
end
exSettings = code;