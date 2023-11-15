function intensitypyramidblock(block)
% Level-2 M file S-function for upsampling image pyramid levels.
  setup(block);
%endfunction

function setup(block)

  %% Register dialog parameter: edge direction
  block.NumDialogPrms = 4;
  %block.DialogPrmsTunable = {'Tunable'};

  %% Register ports
  block.NumInputPorts  = 1;
  block.NumOutputPorts = 1;

  %% Setup port properties
  block.SetPreCompInpPortInfoToDynamic;
  block.SetPreCompOutPortInfoToDynamic;
 % block.InputPort(1).Dimensions   = -1;
  block.InputPort(1).DirectFeedthrough = 1;
  block.InputPort(1).DatatypeID   = 0;
  block.InputPort(1).Complexity   = 'Real';
  block.InputPort(1).SamplingMode = 'Sample';
  block.InputPort(1).Overwritable = true; % No in-place operation


  block.OutputPort(1).Complexity   = 'Real';
  block.OutputPort(1).SamplingMode = 'Sample';

  %% Register block methods (through MATLAB function handles)
  block.RegBlockMethod('Outputs', @Output);
  block.RegBlockMethod('SetInputPortDimensions', @SetInputPortDimensions);
  %block.RegBlockMethod('SetOutputPortDimensions', @SetOutputPortDimensions);
  %block.RegBlockMethod('WriteRTW',@WriteRTW);

  %% Block runs on TLC in accelerator mode.
  %block.SetAccelRunOnTLC(true);

%endfunction


%%
%% Block Output method: Perform pyramid level extraction
%%
function Output(block)
  levels = block.DialogPrm(3).Data; % level zero rows

  block.OutputPort(1).Data = IntensityPyramid(block.InputPort(1).Data, levels);


%endfunction


function SetInputPortDimensions(block, idx, di)
    %dims = size(block.OutputPort(1).Data);
    %display(['intensitypyramid: setting input port dimensions, input ' num2str(idx)]);
    %display(di)
    sr = di(1); %block.DialogPrm(1).Data; % source image rows
    sc = di(2); %block.DialogPrm(2).Data; % source image cols
    sl = block.DialogPrm(3).Data; % pyramid levels
    bo = block.DialogPrm(4).Data; % Offset of base image from original input image

    [minV ss] = min([sr sc]);
    %[maxV ls] = max([sr sc]);
    %display(['intensitypyramid level: ' num2str(sr) ' by ' num2str(sc) '; level = ' num2str(sl) ' offset = ' num2str(bo)])
    [rows cols] = PyramidLevelSizes(sr, sc, bo+sl);
    inRows = rows(bo+1);
    inCols = cols(bo+1);
    %display(['intensitypyramid: inRows = ' num2str(inRows) ' inCols = ' num2str(inCols)])
    block.InputPort(idx).Dimensions = di; %[rows(bo+1) cols(bo+1)];
    rows = rows((bo+1):end);
    cols = cols((bo+1):end);
    if ss == 1
        row = rows(1) + rows(2) + 1;
        col = sum(cols(2:end));
        col = max(col,sc);
    else
        row = sum(rows(2:end));
        row = max(row,sr);
        col = cols(1) + cols(2) + 1;
    end
    %display(['intensitypyramid level: out rows = ' num2str(row) ' out cols = ' num2str(col)])
    %pause

    block.OutputPort(idx).Dimensions = [row col];
%endfunction
