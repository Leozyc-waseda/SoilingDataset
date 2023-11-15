function varargout = MarkupMenu(varargin)
% MARKUPMENU M-file for MarkupMenu.fig
%      MARKUPMENU, by itself, creates a new MARKUPMENU or raises the existing
%      singleton*.
%
%      H = MARKUPMENU returns the handle to a new MARKUPMENU or the handle to
%      the existing singleton*.
%
%      MARKUPMENU('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MARKUPMENU.M with the given input arguments.
%
%      MARKUPMENU('Property','Value',...) creates a new MARKUPMENU or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MarkupMenu_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MarkupMenu_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MarkupMenu

% Last Modified by GUIDE v2.5 29-Mar-2007 14:48:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @MarkupMenu_OpeningFcn, ...
    'gui_OutputFcn',  @MarkupMenu_OutputFcn, ...
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


% --- Executes just before MarkupMenu is made visible.
function MarkupMenu_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   commands line arguments to MarkupMenu (see VARARGIN)

% Choose default commands line output for MarkupMenu
handles.output = hObject;
handles.files = [];
% Update handles structure
guidata(hObject, handles);
addpath(cd);

% UIWAIT makes MarkupMenu wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the commands line.
function varargout = MarkupMenu_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default commands line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (isempty(handles.files))
    errordlg('You must select some files');
elseif(isempty(get(handles.eventNumber,'String')))
    errordlg('You must enter an event number');
else
    if (isempty(get(handles.commands,'String')))
        eval(['sacPlots(handles.files,',get(handles.eventNumber,'String'),');']);
    else
        eval(['sacPlots(handles.files,',get(handles.eventNumber,'String'),...
            ',',get(handles.commands,'String'),');']);
    end
end


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

p = cd;
if (exist('data.mat'))
    load('data.mat');
    cd(pathname);
end
filename = uipickfiles;
cd(p);
if (isequal(filename,0))
    slideOK;
    filename = [];
else
    f = findstr(filename{1},'\');
    if (isempty(f))
        f = findstr(filename{1},'/');
    end
    pathname = filename{1};
    pathname(f(end)+1:end) = [];
    save('data.mat','pathname');
end
handles.files = filename;
%Update handles structure
guidata(hObject, handles);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if (isempty(handles.files))
    errordlg('You must select some files');
else
    if (isempty(get(handles.commands,'String')))
        eval(['markEye(handles.files);']);
    else
        eval(['markEye(handles.files,',get(handles.commands,'String'),');']);
    end
end


% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over pushbutton4.
function pushbutton4_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)




function commands_Callback(hObject, eventdata, handles)
% hObject    handle to commands (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of commands as text
%        str2double(get(hObject,'String')) returns contents of commands as a double

% --- Executes during object creation, after setting all properties.
function commands_CreateFcn(hObject, eventdata, handles)
% hObject    handle to commands (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)





function eventNumber_Callback(hObject, eventdata, handles)
% hObject    handle to eventNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of eventNumber as text
%        str2double(get(hObject,'String')) returns contents of eventNumber as a double


% --- Executes during object creation, after setting all properties.
function eventNumber_CreateFcn(hObject, eventdata, handles)
% hObject    handle to eventNumber (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end




% --- Executes on button press in plottrace.
function plottrace_Callback(hObject, eventdata, handles)
% hObject    handle to plottrace (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if (isempty(handles.files))
    errordlg('You must select some files');
else
    if (findstr(version,'7'))
        [filez,glob] = strip_file_path(handles.files);
        for ii = 1:length(filez)
            fnam = filez{ii};
            fnam = [glob,fnam];
            [data,pupil] = loadCalibTxt(fnam); %load the file and clean up a bit
            %make some plots in a sliding window
            sf = get(handles.commands,'String');
            eval(['sf = parseinputs(',['{',sf,'}'],');']);
            sf = getvalue('sf',sf);
            h = slidingPlot(data(1:3,:),zeros(1,length(data)),...
                zeros(1,length(data)),sf);
            drawnow;pause;close(h);
        end
    else
        errordlg('Display is not backward compatible with this version of Matlab');
    end
end
