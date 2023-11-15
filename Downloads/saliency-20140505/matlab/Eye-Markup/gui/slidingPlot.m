function varargout = slidingPlot(varargin)
% SLIDINGPLOT M-file for slidingPlot.fig
%      SLIDINGPLOT, by itself, creates a new SLIDINGPLOT or raises the existing
%      singleton*.
%
%      H = SLIDINGPLOT returns the handle to a new SLIDINGPLOT or the handle to
%      the existing singleton*.
%
%      SLIDINGPLOT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SLIDINGPLOT.M with the given input arguments.
%
%      SLIDINGPLOT('Property','Value',...) creates a new SLIDINGPLOT or raises the
%      existing singleton*.  Starting from the left, property valuetxt pairs are
%      applied to the GUI before slidingPlot_OpeningFunction gets called.  An
%      unrecognized property name or invalid valuetxt makes property application
%      stop.  All inputs are passed to slidingPlot_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help slidingPlot
% May be broken due to modification in plotcolor
% Last Modified by GUIDE v2.5 19-Feb-2007 20:48:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @slidingPlot_OpeningFcn, ...
    'gui_OutputFcn',  @slidingPlot_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before slidingPlot is made visible.
function slidingPlot_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to slidingPlot (see VARARGIN)

% Choose default command line output for slidingPlot
handles.output = hObject;

handles.vel = varargin{2};
handles.r = varargin{3};
handles.sf = varargin{4};
handles.data = varargin{1};
if iscell(handles.data)
    if isstr(handles.data{1})
        handles.data = load(handles.data{1});
        handles.data = handles.data(:,1:3)';
    end
end
handles.time = (0:length(handles.data)-1) .* (1000./handles.sf);


set(findobj('Tag','maxvalue'),'String',['Max: ',num2str(handles.time(end))]);
set(findobj('tag','leftslide'),'String',1);
set(findobj('tag','rightslide'),'String',num2str(handles.time(end)));
set(findobj('Tag','slider'),'Min',1);
set(findobj('Tag','slider'),'Max',size(handles.data,2));
set(findobj('Tag','slider'),'SliderStep',[1./size(handles.data,2) 1./size(handles.data,2)]);
set(findobj('Tag','slider'),'Value',1);
%set(findobj('Tag','valuetext'),'String',get(findobj('Tag','slider'),'Value'));
set(findobj('Tag','valuetext'),'String',num2str(0));
set(findobj('Tag','editwindow'),'String',num2str(handles.time(end)));

% Update handles structure
guidata(hObject, handles);

%figure(2);
% UIWAIT makes slidingPlot wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = slidingPlot_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on slider movement.
function slider_Callback(hObject, eventdata, handles)
% hObject    handle to slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
pos  = get(hObject,'Value');
win1 = str2num(get(findobj('tag','editwindow'),'String'));
%convert to points

win1 = win1/(1000/handles.sf);
win2 = win1;
win1 = floor(pos - win1);
win2 = ceil(pos + win2);
if ( win1 < 1)
    win1 = 1;
end
if (win2 > size(handles.data,2))
    win2 = size(handles.data,2);
end
%put into ms
w1 = win1 * (1000/handles.sf);
w2 = win2 * (1000/handles.sf);
set(findobj('tag','valuetext'),'String',num2str(w1));
set(findobj('tag','rightslide'),'String',num2str(w2));
f =findobj('type','axes');
set(f(1:end-1),'XLim',[w1 w2]);
set(gcf,'CurrentAxes',handles.xypos);
cla;
plotcolor(handles.xypos,handles.time(:,win1:win2),handles.data(:,win1:win2));
drawnow;




% --- Executes during object creation, after setting all properties.
function slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end





function editwindow_Callback(hObject, eventdata, handles)
% hObject    handle to editwindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editwindow as text
%        str2double(get(hObject,'String')) returns contents of editwindow as a double
plotTrace(handles,handles.time,handles.data,handles.vel,handles.r, '-', 1);


% --- Executes during object creation, after setting all properties.
function editwindow_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editwindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function editmax_Callback(hObject, eventdata, handles)
% hObject    handle to editmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editmax as text
%        str2double(get(hObject,'String')) returns contents of editmax as a double


% --- Executes during object creation, after setting all properties.
function editmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes during object creation, after setting all properties.
function valuetxt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to valuetxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called




% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to editwindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editwindow as text
%        str2double(get(hObject,'String')) returns contents of editwindow as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editwindow (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function plotTrace(handles,time,data, vel,r, s, c, nfr)
% plot a data trace using style 's'; clear previous plots if 'c'
% given; use a fixed display range unless nfr is given and non-zero
set(gcf,'doublebuffer','on');

if (nargin >= 8 && nfr ~= 0), useFixedRange = 1;
else useFixedRange = 0; end
if (nargin <= 1), s = '-'; end
sz = size(data);

set(gcf,'CurrentAxes',handles.xypos);
if (nargin >= 7 && c ~= 0), hold off; end
plotcolor(handles.xypos,time,data); ylabel('Pupil X/Y');
set(gca,'xlim',[0 640]);
set(gca,'ylim',[0 480]);
grid on; hold on;

set(gcf,'CurrentAxes',handles.xpos);
if (nargin >= 7 && c ~= 0), hold off; end
plotcolor(handles.xpos,time,data([1 3], :)); ylabel('Pupil X');
if (useFixedRange ~= 0), axis([ 1 sz(2) 0 512]); end
set(gca, 'XTickLabel', []); grid on; hold on;

set(gcf,'CurrentAxes',handles.ypos);
if (nargin >= 7 && c ~= 0), hold off; end
plotcolor(handles.ypos,time,data([2 3],:)); ylabel('Pupil Y');
if (useFixedRange ~= 0), axis([ 1 sz(2) 0 512]); end
set(gca, 'XTickLabel', []); grid on; hold on;

if ((length(vel) ~= 1) && (length(r) ~= 1))
    set(gcf,'CurrentAxes',handles.velpca);
    yy = plotyy(handles.velpca,time,r,time,vel);
    set(get(yy(1),'Ylabel'),'String','min(eigval)/max(eigval)');
    set(yy(1),'ylim',[0 1.2]);
    set(get(yy(2),'Ylabel'),'String','Velocity (Deg/Sec)');
end
xlabel('Time(ms)');

drawnow;