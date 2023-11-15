% //////////////////////////////////////////////////////////////////// %
%           Surprise Linear Model - Copyright (C) 2004-2007            %
% by the University of Southern California (USC) and the iLab at USC.  %
% See http://iLab.usc.edu for information about this project.          %
% //////////////////////////////////////////////////////////////////// %
% This file is part of the iLab Neuromorphic Vision Toolkit            %
%                                                                      %
% The Surprise Linear Model is free software; you can                  %
% redistribute it and/or modify it under the terms of the GNU General  %
% Public License as published by the Free Software Foundation; either  %
% version 2 of the License, or (at your option) any later version.     %
%                                                                      %
% The Surprise Linear Model is distributed in the hope                 %
% that it will be useful, but WITHOUT ANY WARRANTY; without even the   %
% implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      %
% PURPOSE.  See the GNU General Public License for more details.       %
%                                                                      %
% You should have received a copy of the GNU General Public License    %
% along with the iBaysian Surprise Matlab Toolkit; if not, write       %
% to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,   %
% Boston, MA 02111-1307 USA.                                           %
% //////////////////////////////////////////////////////////////////// %
%
% Primary maintainer for this file: T. Nathan Mundhenk <mundhenk@usc.edu>
% $Revision: 55 $ 
% $Id$
% $HeadURL: https://surprise-mltk.svn.sourceforge.net/svnroot/surprise-mltk/source/surprise_toolkit/example_graph.m $

global BEST_VAL;
BEST_VAL = 10;
global LOC_BEST_VAL; 
LOC_BEST_VAL = 10;
global BESTFILE;
BESTFILE = '/lab/mundhenk/linear-classifier/log/simplex-bestfile.txt';
global OUTERLOOPS;
OUTERLOOPS = 1;
global RESULTS;
RESULTS = struct('description','Holds results to be passed around');
global ERRORRECORD;
ERRORRECORD = '/lab/mundhenk/linear-classifier/log/error-record.txt';
global ERRORFILE;
ERRORFILE = '/lab/mundhenk/linear-classifier/log/error-file.txt';
global DEBUGFILE;
DEBUGFILE = '/lab/mundhenk/linear-classifier/log/debug-file.txt';
global EOPTIONS;
EOPTIONS = struct('description','Holds extra options to be passed around');
global LOGFILE;
LOGFILE = '/lab/mundhenk/linear-classifier/log/simplex-log.txt';
global COUNT;
COUNT = 1;
global ERROR_STATE;
ERROR_STATE.regression           = struct('description','holds error state durring regression');
ERROR_STATE.regression.isError   = 0;

EOPTIONS.UseRandom      = 2;
EOPTIONS.RandSpread     = 5;
EOPTIONS.RandOffset     = 0;
EOPTIONS.RandKeep       = 0.333;
EOPTIONS.ForceCataclism = 0;
EOPTIONS.debugLevel     = 2;
EOPTIONS.useUpperBound  = 'yes';
EOPTIONS.upperBound     = [ 8 ; 8 ; 8 ];
EOPTIONS.useLowerBound  = 'yes';
EOPTIONS.lowerBound     = [ 0.01 ; 0.01 ; 0.01];

% Init parameters
Options = linear_model_get_optim_set();
flog = fopen(Options.prec_out_log,'w');
fclose(flog);
flog = fopen(Options.prec_in_log,'w');
fclose(flog);
%X0 = [2.7516 6.1991 2.5 3 9.899 2.8148 7];

fid = fopen(LOGFILE,'w');
fclose(fid);

X = Options.start;

% Error = 0.2349999999999999866773237044981215149164199829101562500000000000
X = [ 0.8373879360465116272749241943529341369867324829101562500000000000  ...
      0.3126931814916846796847949008224532008171081542968750000000000000  ...
      1.5666999999999999815258888702373951673507690429687500000000000000 ];

% Error = 0.2570000000000000062172489379008766263723373413085937500000000000
X = [ 0.0463315003831105479137342229023488471284508705139160156250000000 ...	
      4.9189681327151015821641522052232176065444946289062500000000000000 ...	
      4.6095565769071136230650154175236821174621582031250000000000000000 ];

% Error = 0.257000000000000006
X = [ 0.0463315003831105479137342229023488471284508705139160156250000000 ; ...	
      4.9189681327151015821641522052232176065444946289062500000000000000 ; ...	
      4.6095565769071136230650154175236821174621582031250000000000000000 ; ]; 

% Error = 0.250500000000000000
X = [ 3.3114760187666409940732137329177930951118469238281250000000000000 ; ...
      6.9729893775717002313285775016993284225463867187500000000000000000 ; ...
      4.7586313207968853333795777871273458003997802734375000000000000000 ];

%X = [ 1 1 0.5 3];

A = [];
b = [];
Aeq = [];
Beq = [];
%ub = [10,10,10,10,20,10,10];
%lb = [0,0,0,0,0,0,0];
ub = Options.ub;
lb = Options.lb;
NONLCON = [];

methodType = 'nfminsearch';

% Clean log
command = ['rm -f ' Options.meta_log];
unix(command);
% Store info in case we crash and need to start again
time = clock;
command = ['echo STARTING: METHOD ',methodType,' ',date,' ',...
        num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
        ' >> ' Options.meta_log];
unix(command);

options = [];

%doThis = @linear_model_fmincon;
doThis = @linear_model_fmincon_svm;
[x,fval,exitflag,output] = nfminsearch2(doThis,X,options);
