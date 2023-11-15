function data = exportEyeData(data, params, ofname)
% This function saves and exports the data to an output file. 
% Output to the file is given in rows, one per sample, with the
% fields requested in the stats parameter.  This file is meant to
% be read by ezvision but can be read by other files.
%
% Four lines are written in the header:
% period = #Hz 
% ppd = #
% trash = #
% cols = <stats>
% with successive rows as the stats.
% 
% The output simply passes on the data.
% For a list of possible statistics look at the export
% configuration file export.mconf. 
%
% written: Nov 2009
% author : John Shen
% iLab - University of Southern California
%*****************************************************************

% TODO: validate data more strictly
if ~isstruct(data) 
    warning('exportEyeData:noData','No data to process...');
    return;
end

autosave = getvalue('autosave',params);
% give autosave choice
if autosave == 0
    reply=input(['Save trace to ' ofname '? (RET=yes)'], 's');
    if ~(isempty(reply) || lower(reply(1)) == 'y')
        return;
    end
end

sf = getvalue('sf', params);
ppd = getvalue('ppd', params);
trash = getvalue('trash', params);
vblevel = getvalue('verbose',params);
out_stats_on_ev=getvalue('events_to_stat',params);
give_col_header = getvalue('write_col_header',params);
if(isempty(trash)), trash = 0; end

% list of stats
stats = getvalue('stats', params);
conf_file = getvalue('export_conf_file', params);
marker = getvalue('event_stat_marker',params); % usually '*';
exportSettings = readExportConfig(conf_file);

% open output file
[fil message] = fopen(ofname, 'w');
if(~isempty(message))
  error('importEyeData:FileNotOpened', ... 
	'File ''%s'' can not be opened because: %s', ofname, message);
end

% write period,ppd,trash first - takes on role of addmetadata
% TODO: option this or make a new sub in pipeline
fprintf(fil, 'period = %dHz\n', sf);
fprintf(fil, 'ppd = %.1f\n', max(ppd)); % this should be revised;
					% currently does not the different
					% ppd in x and y
fprintf(fil, 'trash = %d\n', trash);

% this messy bit determines how we output the data
index_Stat = NaN(1,length(stats));
titles = cell(1,length(stats));

for ii = 1:length(stats)
  % find correct stat index
  foo = strmatch(stats{ii},{exportSettings.key});
  if isempty(foo)
    for jj = 1:length(exportSettings)
      test = strmatch(stats{ii}, exportSettings(jj).aliases);
      if ~isempty(test)
    	foo = jj;
        break; 
      end
    end
  end

  % if the stat is not in the conf file and is not a field
  if isempty(foo)
    if isfield(data,stats{ii})
       % graft the new field data on with defaults (hard-coded)
       % FIXME: this should be made more understandable
      newStat = struct('aliases', {stats{ii}}, 'format', '%.1f',...
		       'is_on_event', 0, 'key', stats{ii}, ...
		       'reference', stats{ii});
      exportSettings(end+1) = newStat; %#ok<AGROW>
      foo = length(exportSettings);
    elseif isfield(data.events,stats{ii})
     newStat = struct('aliases', {stats{ii}}, 'format', '%.1f',...
		       'is_on_event', 1, 'key', stats{ii}, ...
		       'reference', stats{ii});
      exportSettings(end+1) = newStat; %#ok<AGROW>
      foo = length(exportSettings);
        
    else
        fclose(fil);
        error('exportEyeData:badOpt', 'Unknown statistic %s', ...
	    stats{ii});
    end   
  end
  index_Stat(ii) = foo;
  
  titles{ii} = exportSettings(index_Stat(ii)).key;
  % add * to the titles of stats with event-only data
  if exportSettings(index_Stat(ii)).is_on_event
    titles{ii} = [marker titles{ii}];
  end
end 

% output header - write cols line:
if(give_col_header)
  fprintf(fil, 'cols = %s\n', strjoin(titles));
end

output = zeros(length(stats),data.len);
sformat = [];

% tabulate the data according to the stats string array
% NB: right now ezvision can only take in scalar data (to be fixed)
for ii = 1:length(stats)
  row = zeros(1,data.len); % should always be a row 
    
  % find correct stat
  exSuite = exportSettings(index_Stat(ii));
  
  if ~exSuite.is_on_event % for all points
    % can we directly grab data with a fxn?
    if isa(exSuite.reference,'function_handle')
      row = exSuite.reference(data);
    else % dynamic field reference
      row = data.(exSuite.reference);
    end
  else
    % grab the events we want to analyze
    is_markworthy = ismember(cell2mat({data.events.type}), ...
			     out_stats_on_ev);
    event_seld = data.events(is_markworthy);
    
    if ~isempty(event_seld) % there must be interesting events
        % if we need to index into them
        time_sel = cell2mat({event_seld.onset});

        % can we directly grab data with a fxn?
        if isa(exSuite.reference,'function_handle')
        row(time_sel) = exSuite.reference(event_seld);
        else % dynamic field reference
        row(time_sel) = cell2mat({event_seld.(exSuite.reference)});
        end
    end
  end

  output(ii,:) = row;

  % build output format string
  sformat = strcat(sformat, [' ' exSuite.format]);
end

% using output format string, dump data to file
fprintf(fil, [sformat '\n'], output);

% close up
fclose(fil);
if vblevel >= -1 
  fprintf('Wrote %s...\n\n', ofname);
end
