function doit(name, scrw, scrh, varargin)
% DOIT(name)
% function doit(name, scrw, scrh)
%
% name: takes in a .mat file with ISCAN and PSYCH values
% Optional parameters scrw and scrh" the viewing dimensions
%
% Saves a data structure with calibrated eye traces and writes for
% each event an .eye file with the following structure for each line:
%   PORx PORy diam 0.0
% Also writes header if special processing is needed, in the form:
%   # field val

if (nargin < 2) %default parameters
     scrw = 1920;
     scrh = 1080;
end

load(name);

% check to make sure there are an equal number of sessions
numI = length(ISCAN.data);
numP = length(PSYCH.fname);
if (numI ~= numP)
  error('doit:badMAT', '%d ISCAN vs %d PSYCH sessions', numI, numP)
end

run_inter = ismember('interactive',varargin);
drift_corr = ismember('drift correct', varargin);
run_auto = ismember('auto', varargin);

fprintf('Settings: run_inter = %d, drift_corr = %d, run_auto = %d\n',...
        run_inter, drift_corr, run_auto);
    
% run main program
if run_inter
    CALIB = analyzeMovieExpInter(ISCAN, PSYCH, scrw, scrh,drift_corr);
elseif run_auto
  CALIB = analyzeMovieExpAuto(ISCAN, PSYCH, scrw, scrh);
else
    CALIB = analyzeMovieExp(ISCAN,PSYCH, scrw, scrh);
end
% save data in separate .eye files
saveCalibTxt(CALIB);

% save data as a matlab file
[itspath,itsname,itsext] = fileparts(name);
outName = [itspath,itsname,'-CALIB.mat'];
save(outName,'CALIB');