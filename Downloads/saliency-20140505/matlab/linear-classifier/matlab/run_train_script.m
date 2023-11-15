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

global COUNT;
COUNT = 1;

global ERROR_STATE;
ERROR_STATE = [];

% Init parameters
Options = linear_model_get_optim_set();
flog = fopen(Options.prec_out_log,'w');
fclose(flog);
flog = fopen(Options.prec_in_log,'w');
fclose(flog);
%X0 = [2.7516 6.1991 2.5 3 9.899 2.8148 7];


% Chan Train
%X = Options.start;
%X = [0.8333 0.4167 1.6667];  
% Error = 0.2349999999999999866773237044981215149164199829101562500000000000
%X = [ 0.8373879360465116272749241943529341369867324829101562500000000000  ...
%      0.3126931814916846796847949008224532008171081542968750000000000000  ...
%      1.5666999999999999815258888702373951673507690429687500000000000000 ];

% From simplex train
% Error = 0.222000000000000003
%X = [ 0.7011478635474999165921872190665453672409057617187500000000000000  ...
%      0.3106499463307507702403142957336967810988426208496093750000000000  ...
%      1.4764179753326787114531271072337403893470764160156250000000000000 ];
  
% Error = 0.2179999999999999993338661852249060757458209991455078125000000000
%X = [ 0.8773092344262305442015303924563340842723846435546875000000000000 ...
%      0.3127783654225734788489887705509318038821220397949218750000000000 ...
%      1.5660499968021082128899479357642121613025665283203125000000000000 ];

% Target Training  
% Error = 0.2959999999999999853450560749479336664080619812011718750000000000
%X = [ 0.7773092344262305664059908849594648927450180053710937500000000000 ...	
%      0.1000000000000000000000000000000000000000000000000000000000000000 ...	
%      0.7915187780899255454158947031828574836254119873046875000000000000 ];

% Error = 0.2580000000000000071054273576010018587112426757812500000000000000
%X = [ 0.0454107428933686260719149174747144570574164390563964843750000000 ...	
%      4.9188118910252294213591994775924831628799438476562500000000000000 ...	
%      4.6095566781943455580972113239113241434097290039062500000000000000 ]; 

% Error = 0.2570000000000000062172489379008766263723373413085937500000000000
%X = [ 0.0463315003831105479137342229023488471284508705139160156250000000 ...	
%      4.9189681327151015821641522052232176065444946289062500000000000000 ...	
%      4.6095565769071136230650154175236821174621582031250000000000000000 ];
  
% Error = 0.233500000000000013 ErrorRecord 0.237500  0.233500  0.241000  0.245500
%X = [ 3.4751354836940127057687277556397020816802978515625000000000000000 ...
%      6.3745207398689851530093619658146053552627563476562500000000000000 ...
%      4.1360488529659154011142163653858006000518798828125000000000000000 ];

% Chan Train 'UHIOLTWXE_train'
X = [ 0.8773092344262305442015303924563340842723846435546875000000000000 ...
      0.3127783654225734788489887705509318038821220397949218750000000000 ...
      1.5660499968021082128899479357642121613025665283203125000000000000 ...
      0.0001 ...
      0.0001 ...
      0.0001 ...
      0.0001 ...
      0.0001 ];
  
% Chan Train 'UHIOLTXE_train'
X = [ 0.8773092344262305442015303924563340842723846435546875000000000000 ...
      0.3127783654225734788489887705509318038821220397949218750000000000 ...
      1.5660499968021082128899479357642121613025665283203125000000000000 ...
      0.1 ...
      0.1 ...
      0.1 ...
      0.1 ];
  
%Fac Train
%X = [ 0.1 0.1 0.5 3];

A = [];
b = [];
Aeq = [];
Beq = [];
ub = [ 2, 2, 2, 2, 2, 2, 2];
lb = [ 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001];
%ub = Options.ub;
%lb = Options.lb;
NONLCON = [];

methodType = 'fmincon';

% Clean log
command = ['rm -f ' Options.meta_log];
unix(command);
% Store info in case we crash and need to start again
time = clock;
command = ['echo STARTING: METHOD ',methodType,' ',date,' ',...
        num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
        ' >> ' Options.meta_log];
unix(command);

%options = optimset('DiffMinChange',0.1);
%options = optimset('DiffMinChange',0.05,'GradObj','on','LargeScale','off','TolX',1e-32,'TolFun',1e-32);
options = optimset('DiffMinChange',0.001,'TolX',1e-32,'TolFun',1e-32);
doThis = @linear_model_fmincon_svm;
[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(doThis,X,A,b,Aeq,Beq,lb,ub,NONLCON,options);
