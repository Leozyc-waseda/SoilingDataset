function upsamplelevel(block)
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
  sr = block.DialogPrm(1).Data; % level zero rows
  sc = block.DialogPrm(2).Data; % level zero cols
  cLevel = block.DialogPrm(3).Data;
  sLevel = block.DialogPrm(4).Data;

  tmp = UpsamplePyrLevel(block.InputPort(1).Data, cLevel, sLevel, sr, sc);
  %[tr tc] = size(tmp);
  %display('output port dimensions')
  %display(block.OutputPort(1).Dimensions)
  %class(block.OutputPort(1).Dimensions)
  %block.OutputPort(1).Data = -10000*ones(block.OutputPort(1).Dimensions);
  block.OutputPort(1).Data = tmp;


%endfunction


function SetInputPortDimensions(block, idx, di)
    %dims = size(block.OutputPort(1).Data);
    %display(['setting input port dimensions, input ' num2str(idx)]);
    %display(di)
    block.InputPort(idx).Dimensions = di;
    sr = block.DialogPrm(1).Data;
    sc = block.DialogPrm(2).Data;
    sl = block.DialogPrm(3).Data;
    %display(['upsample level: ' num2str(sr) ' by ' num2str(sc) '; level = ', num2str(sl)])
    [row col] = PyramidLevelDims(sr, sc, sl);
    %display(['upsample level: out rows = ' num2str(row) ' out cols = ' num2str(col)])
    %pause

    block.OutputPort(idx).Dimensions = [row col];
%     switch idx
%         case 1
%             block.InputPort(1).Dimensions = [499 500];
%         case 2
%             block.InputPort(2).Dimensions = 1;
%     end
%endfunction
