function [eyedata,newparams] = importEyeSData(fname,params)
% This function imports eyeS style datainto a proper structure 
% The input file should be like the output which is sent from exportEyeData
% in this markup file.  It does not read any markup which is already done.  
% In the case of Eyelink files, 

% fields of the returned eyedata structure are
%  len
%  xy = 2xlen eye position  
%  pd = 1xlen integer
%  vel = 1Xlen (not read from file)
%  status = 1xlen integer 
% newparams is a cell array which should be passed to parseinputs

% For the header, the period, ppd, and trash are read in first; then,
% comments can be added with a # at the beginning of the line; the last
% line of the header is the specification of cols.

%written: Nov 2009
%author : John Shen
%iLab - University of Southern California
%**************************************************************************

if(nargin == 1) 
    params = defaultparams; 
end
    
vblevel = getvalue('verbose',params);
V = getvalue('verboselevels',params);

%%% open file
[fid message] = fopen(fname, 'r');
if(~isempty(message))
  error('importEyeData:FileNotOpened', ... 
	'File ''%s'' can not be opened because: %s', fname, message);
end

%%% TODO: maybe make these headers less hard-coded

%%% read file headers with format as 
% period = 
% ppd = 
% trash = 
% # (comments, could have many of them)
% cols = 
% and put in new setting structure
sf = cell2mat(textscan(fid,'period = %fHz'));
ppd = cell2mat(textscan(fid,'pd = %f'));
trash = cell2mat(textscan(fid,'trash = %d'));
% ignore comments in headers until seeing cols
goflag=1;
while goflag
    l = fgetl(fid);
    if strcmpi(l(1), '#')
        % just comments, ignore
    elseif strcmpi(l(1:4), 'cols')
        % parse cols and then exit the loop
        cols = regexp(l, ' \*?', 'split');
        cols(1:2) = [];
        goflag = 0;
    else
        % exit loop if seeing something other than # or cols
        goflag = 0;
    end
end
newparams = parseinputs(params, 'sf', sf, 'ppd', ppd, 'trash', trash, 'stats', cols);

%%% import the rest of data
rawdata = textscan(fid, ''); % reads data by space-delimed columns 
			      % into a 1xC cell array of row matrices
n_fields = length(rawdata); 
if n_fields < 2
  error('importEyeData:BadEyeFile',...
	'Not enough fields in file %s or improper comment', fname);  
end

%%% populate data
eyedata.len = length(rawdata{1});
if eyedata.len == 0
  error('importEyeSData:FileNotRead',...
	'File %s is not readable', fname);  
end

% fill data structure with row matrices
% this is where this differs from importEyeData;
eyedata.xy = [rawdata{1:2}]'; % in screen coordinates
eyedata.pd = rawdata{3}';
eyedata.vel = getVel(eyedata,newparams);
eyedata.status = rawdata{4}';

% report
if vblevel>=V.SUB
    fprintf('\t%d samples loaded into data\n', eyedata.len);
end