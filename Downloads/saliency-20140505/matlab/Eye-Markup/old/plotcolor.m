function plotcolor(data,request,linestyle,do_connect)
%function plotcolor(g,time,x,corder);
% 0 'blue' = fixation
% 1 'green' = in saccade
% 2 'red' = in blink
% 3 'yellow' = in blink during a saccade
% 4 'magenta' = smooth pursuit
% 5 'black' = unmarked
% 6 'cyan' = saccade combined

% g is figure gca
% time is for if x is in [x status]
% x is data (either [x y status] or [x status]

corder = {'b','g','r','y','m','k','c'};
% add the linestyle
% corder = cellfun(@(x)horzcat(x,linestyle),corder,'UniformOutput',false);

NaNstyle = 'k:';
if nargin < 4
    do_connect = true;
end
if nargin < 3
    linestyle = '-';
else
    linestyle = linestyle(end); % pick off only the marker style
end
if nargin < 2
    request = 'xy';
end

% default is that x-axis is time (in samples)
xdat = 1:data.len;
switch request
    case 'xy'
        xdat = data.xy(1,:);
        ydat = data.xy(2,:);
    case 'x'
        ydat = data.xy(1,:);
    case 'y'
        ydat = data.xy(2,:);
    otherwise
        if(isfield(data,request))
            ydat = data.(request); %by field
        else
            ydat = data.status; %by default
        end
end

hold on;
for ii = 1:numel(code_range)
    i_stat = code_range(ii);
    [eb,ee] = getBounds(data.status==i_stat);
    for jj = 1:length(eb) 
        % for connection purposes we have to plot each region separately
        if do_connect
            range = eb(jj):ee(jj)+1;
        else
            range = eb(jj):ee(jj);
        end
        range(range>data.len) = [];
        
        if do_connect
            plot(xdat(range),ydat(range), [corder{ii} linestyle], ...
                'markersize',2);
        else % small provision for the status code window
            plot(xdat(range),ydat(range), [corder{ii} linestyle], ...
                'LineWidth',4);
        end
    end
end

is_NaNstat = isnan(data.status);
if any(is_NaNstat)
    plot(xdat(is_NaNstat), ydat(is_NaNstat), NaNstyle);
end

hold off;