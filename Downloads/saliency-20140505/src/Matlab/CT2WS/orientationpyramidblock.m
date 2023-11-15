function orientationpyramidblock(block)
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
  angle = block.DialogPrm(4).Data;
  block.OutputPort(1).Data = OrientationPyramid(block.InputPort(1).Data, levels, angle);


%endfunction


function SetInputPortDimensions(block, idx, di)
    %display(['orientationpyramid: setting input port dimensions, input ' num2str(idx)]);
    %dims = size(block.OutputPort(1).Data);
    %display(di)
    block.InputPort(idx).Dimensions = di;
    sr = block.DialogPrm(1).Data;
    sc = block.DialogPrm(2).Data;
    sl = block.DialogPrm(3).Data;
    [minV ss] = min([sr sc]);
    %[maxV ls] = max([sr sc]);
    %display(['orientationpyramid level: ' num2str(sr) ' by ' num2str(sc) '; level = ', num2str(sl)])
    [rows cols] = PyramidLevelSizes(sr, sc, sl);
    if ss == 1
        row = rows(1) + rows(2) + 1;
        col = sum(cols(2:end));
        col = max(col,sc);
    else
        row = sum(rows(2:end));
        row = max(row,sr);
        col = cols(1) + cols(2) + 1;
    end
    %display(['orientationpyramid level: out rows = ' num2str(row) ' out cols = ' num2str(col)])
    %pause

    block.OutputPort(idx).Dimensions = [row col];
%endfunction
