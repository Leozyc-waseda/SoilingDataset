function [data, args] = runPipeline(func_stack, data, args)
% RUNPIPELINE runs a function stack on data
%   currently, an import function must be the first function
% 
% NB: matlabR2007a doesn't understand that we're trying to help it by doing
% two versions of try catch here, and mistakenly throws an error instead of
% a warning - comment out the matlabVersion >= 7.5 code if this is a
% problem.

if isempty(func_stack);
    disp('Nothing to do!');
    return;
end

vblevel = getvalue('verbose',args);
V = getvalue('verboselevels',args);
code_nums = getvalue('code_nums',args);
scode = getvalue('code',args);

%%% main loop %%%
for ii = 1:length(func_stack)
    fh = func_stack{ii}; % get the function handle

    func_name = getReferent(fh);
    if (vblevel >= V.TOP) && ~strcmp('fprintf',func_name)
        fprintf('Running function %s...\n', func_name);
    end
    
    % TODO: is this really necessary?
    if matlabVersion>=7.5 % try catch is older in matlab R2007a and earlier,
        % hence this dbl block
        try
            [data,args] = runHandle(fh,data,args);
            data.vel = getVel(data,args); % update velocity
            data.events = updateStats(data, args); %update events
        catch myException
            fprintf('Error tripped in function %s or update function: \n%s\nSkipping file...\n', ...
                func_name,myException.message);
            rethrow(myException);
        end
    else % old matlab try/catch syntax
        try
            [data,args] = runHandle(fh,data,args);
            data.vel = getVel(data,args);
            data.events = updateStats(data, args);
        catch
            fprintf('Error tripped in function %s or update function: Skipping file...\n', ...
                func_name);
        end
    end
    
    % debugging: show progress
    if vblevel >= V.PLOT
        figure(1);
        %yset(gcf,'Position', [568 225 665 743]);
        plotTrace(data, args, 'k-', true, true);      
    end
    
    %some more info for debug level
    if vblevel >= V.DATA,
        % let's see how many events we have right now:
        if length(data.events) == 1
            disp('Events not yet labeled...');
        else
            ev_type = [data.events.type];
            disp('Event list:');
            code_names = fieldnames(scode);
            for jj = 1:numel(code_nums)
                jj_code = code_nums(jj);
                if any(ev_type == jj_code)
                    fprintf('\t %s: %d events labeled\n', code_names{jj},sum(ev_type == jj_code));
                end
            end
        end
    end   % end main (function) loop
    
    if vblevel>=V.USER
        disp('Press any key to continue...');
        pause
    end
end
end

function [data,args] = runHandle(fh, data, args)
% if function is built in or doesn't take a data-args pair,
% run it independently and off the data...
if nargin(fh) == 0 || exist(func2str(fh),'builtin') == 5
    fh(); % just run the function by itself
else
    switch nargout(fh)
        case 2 % usually for input
            [data, args] = fh(data,args);
        case {1, -1} % usually for processing
            data = fh(data,args);
        case 0 % usually for plotting
            fh(data,args);
    end
end
end

function fname = getReferent(fhh)
    if ~isa(fhh,'function_handle')
        error('Not a function handle!')
    end
    fname = func2str(fhh); % get the function name
    
    % get referents of anonymous functions
    pos_anon = regexp(fname, '\@\([\w,]+\)','end');
    if ~isempty(pos_anon)
        pos_argstart = regexp(fname,'[^\@]\([\w,]+\)','start');
        if isempty(pos_argstart) 
            pos_argstart = length(fname);
        end
        fname = fname(pos_anon+1:pos_argstart);
    end
end