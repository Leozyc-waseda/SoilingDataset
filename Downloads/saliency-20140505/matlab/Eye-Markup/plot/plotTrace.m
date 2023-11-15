function plotTrace(data, params, s, c, nfr)
% plot a data trace using style 's'; clear previous plots if 'c'
% given as is nonzero;
% s gives the style (default is line)
% c says whether or not to clear the previous plot (default is no)
% nfr = 0 or not given: use a fixed display range 
%
% This will either plot xy, x, y and pupil
% or color coordinated xy, x, y according to status
%
% prevents flickering when data plots are superimposed
%
% Current status code, number and color legend (subject to change)
% 0 'blue' = fixation
% 1 'green' = in saccade
% 2 'red' = in blink
% 3 'yellow' = in blink during a saccade
% 4 'magenta' = smooth pursuit
% 5 'black' = unmarked
% 6 'cyan' = saccade combined

% matlabR2008b-R2010a in nojvm mode gets upset here 
% when a figure is being created, but figures still plot
% so we turn off those warnings
% NB: plotcolor is now defined inside plotTrace.m

vers = matlabVersion;
nojvmConflict = (~usejava('jvm') && vers >= 7.7 && vers < 8);
   
if nojvmConflict
    oldwarn = warning('off','MATLAB:HandleGraphics:noJVM');
end

set(gcf,'doublebuffer','on');

% set default arguments
if (nargin < 3), s = '-'; end       % default is line style
if (nargin < 4), c = false; end         % default is do not clear
if (nargin < 5)
  useFixedRange = false; 
else 
  useFixedRange = logical(nfr);
end

fixedAxes = getvalue('plot_screenwin', params);

n_figs = 8;
% clear the figure
if c, clf(gcf), end

subplot(n_figs, 1, 1:2);
% plot X/Y
plotcolor(data,'xy',s);
xlabel('Pupil X'); ylabel('Pupil Y');
if (useFixedRange), axis(fixedAxes); end
grid on; hold on;
% fix for low label
% shotgun adjustment b/c subplots are too close together
lbl_pos = get(get(gca,'XLabel'),'Position');
lbl_pos(2) = lbl_pos(2)+50; 
xlabel('Pupil X', 'Position', lbl_pos);

figure(1)
subplot(n_figs, 1, 3);
% plot X
plotcolor(data,'x',s); ylabel('Pupil X');
if (useFixedRange), axis([ 1 data.len fixedAxes(1:2)]); end
set(gca, 'XTickLabel', []);  %x axis label should only
grid on; hold on;            %appear on last graph

subplot(n_figs, 1, 4);
% plot Y
plotcolor(data,'y',s); ylabel('Pupil Y');
set(gca, 'XTickLabel', []);
if (useFixedRange), axis([ 1 data.len fixedAxes(3:4)]); end
grid on; hold on;

subplot(n_figs, 1, 5);
% plot PD
plotcolor(data, 'pd',s); ylabel('Pupil Diam');
grid on; hold on;
set(gca, 'XTickLabel', []);
if (useFixedRange), axis([ 1 data.len 0 800]); end

subplot(n_figs, 1, 6:7)
% plot velocity
plotcolor(data, 'vel','-', true,@semilogy); ylabel('log Pupil vel (deg/s)');
set(gca, 'YTick',10.^(-1:0.5:2.5));
set(gca, 'YTickLabel',-1:0.5:2.5);
set(gca, 'XTickLabel', [] );
grid on; 

% draw saccade threshold
minvel = getvalue('sac_minvel',params);
hold on;
semilogy([1 data.len],[1 1]*minvel,'r--'); 

if (useFixedRange), axis([ 1 data.len 0.09 500]); end

subplot(n_figs, 1, n_figs);
% plot status
plotcolor(data, 'status', s,0); ylabel('Status Code');
set(gca, 'YLim', [-0.5 6.5]);
grid on; hold on;
if (useFixedRange), axis([ 1 data.len -0.5 6.5 ]); end
xlabel('Sample number');

drawnow;

% restore old warning state
if nojvmConflict
    warning(oldwarn)
end

function ploth = plotcolor(data,request,linestyle,do_connect, plotfun)
% function plotcolor(data, request, linestyle, do_connect);
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

ploth = gca;
%NaNstyle = 'k:';
if nargin < 5, plotfun = @plot; end
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

hold off;
for ii = 1:length(code_nums) 
    i_stat = code_nums(ii);
    xdatR = xdat; ydatR = ydat; 
    outRange = (data.status~=i_stat);
    xdatR(outRange) = NaN;
    ydatR(outRange) = NaN;

    if do_connect
      plotfun(xdatR, ydatR, [corder{ii} linestyle], 'markersize',2);
    else
     plotfun(xdatR, ydatR, [corder{ii} linestyle], 'LineWidth',4);
    end
	hold on;
end
hold off;
end
end

