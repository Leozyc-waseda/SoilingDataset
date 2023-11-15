function extractlevel(block)
% Level-2 M file S-function for applying Sobel filtering
% (image edge detection demonstration).
%   Copyright 1990-2005 The MathWorks, Inc.
%   $Revision: 1.1.6.3 $

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

%   block.InputPort(2).Dimensions        = 1;
%   block.InputPort(2).DatatypeID   = 0;
%   block.InputPort(2).Complexity   = 'Real';
%   block.InputPort(2).SamplingMode = 'Sample';
%   block.InputPort(2).Overwritable = false; % No in-place operation
%   block.InputPort(2).DirectFeedthrough = 1;


  %block.OutputPort(1).DatatypeID   = 3;
  block.OutputPort(1).Complexity   = 'Real';
  block.OutputPort(1).SamplingMode = 'Sample';
 % block.OutputPort(1).Dimensions   = [83 125];

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

  tmp = ExtractPyrLevel(block.InputPort(1).Data, block.DialogPrm(3).Data); %block.InputPort(2).Data);
  block.OutputPort(1).Data = tmp;


%endfunction

function SetInputPortDimensions(block, idx, di)
    %dims = size(block.OutputPort(1).Data);
    %display(['extractlevel: setting input port dimensions, input ' num2str(idx)]);
    %display(di)
    sr = block.DialogPrm(1).Data;
    sc = block.DialogPrm(2).Data;
    sl = block.DialogPrm(3).Data;
    bo = block.DialogPrm(4).Data;
    %display(['extractlevel: ' num2str(sr) ' by ' num2str(sc) '; level = ' num2str(sl) 'offset = ' num2str(bo)])
    [rowIn colIn] = PyramidLevelDims(sr, sc, bo);
    [rowOut colOut] = PyramidLevelDims(sr, sc, bo+sl);
    %display(['extractlevel: in rows = ' num2str(rowIn) ' in cols = ' num2str(colIn)])
    %display(['extractlevel: out rows = ' num2str(rowOut) ' out cols = ' num2str(colOut)])
    %pause
    block.InputPort(idx).Dimensions = di;

    block.OutputPort(idx).Dimensions = [rowOut colOut];
%endfunction
