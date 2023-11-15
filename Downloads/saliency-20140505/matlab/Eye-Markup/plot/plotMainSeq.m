function plotMainSeq(evs, params, flds)
    Nevs = numel(evs);
    if nargin < 3
        xfld = 'amp'; yfld = 'pvel'; 
    else
        xfld = flds{1}; yfld = flds{2};
    end
    
    sfld = 'type';
    eb = struct('xy', [evs.(xfld); evs.(yfld)], ...
        'status', [evs.type], 'len', Nevs);
    plotcolor(eb, 'xy', '.', false);
    
    function plotcolor(data,request,linestyle,do_connect)
%function plotcolor(data, request, linestyle, do_connect);
% data: data structure passed from importEyeData 
% (has xy, status, pd, etc. as fields) 
% request: plot the type of data (can be 'xy', 'x', 'y', 'pd, 'status')
% linestyle: the line style  
% do_connect: if true, plot the endpoints together
% colors are now coded in parameters, previous bhvr as follows:
% 0 'blue' = fixation
% 1 'green' = in saccade
% 2 'red' = in blink
% 3 'yellow' = in blink during a saccade
% 4 'magenta' = smooth pursuit
% 5 'black' = unmarked
% 6 'cyan' = saccade combined

% time is for if x is in [x status]
% x is data (either [x y status] or [x status]
corder = getvalue('code_color',params);
scode = getvalue('code',params);
code_nums = getvalue('code_nums',params);

%NaNstyle = 'k:';
if nargin < 4, do_connect = true; end
if nargin < 3, 
    linestyle = '-'; 
else
    linestyle = linestyle(end); % pick off only the marker style
end
if nargin < 2, request = 'xy'; end

% default is that x-axis is time (in samples)
xdat = 1:data.len;
switch request
    case 'xy'
        xdat = getX(data.xy);
        ydat = getY(data.xy);
    case 'x'
        ydat = getX(data.xy);
    case 'y'
        ydat = getY(data.xy);
    otherwise
        if(isfield(data,request))
            ydat = data.(request); %by field
        else
            ydat = data.status; %by default
        end
end

hold on;
for ii = 1:length(code_nums) 
    i_stat = code_nums(ii);
    [eb,ee] = getBounds(data.status==i_stat);
    for jj = 1:length(eb)
        if do_connect
            range = eb(jj):ee(jj)+1;
            range(range>data.len) = [];
            plot(xdat(range),ydat(range), [corder{ii} linestyle], ...
                'markersize',2);
        else % small provision for status codes / discrete values             
            range = eb(jj):ee(jj);
            range(range>data.len) = [];
            plot(xdat(range),ydat(range), [corder{ii} linestyle], ...
                'LineWidth',4);
        end       
    end
end

% this should no longer occur
%is_NaNstat = isnan(data.status);
%if any(is_NaNstat)
%    plot(xdat(is_NaNstat), ydat(is_NaNstat), NaNstyle);
%end

hold off;
end
end