%NPclassify_fmincon
%
% This will attempt to minimize the error from NP-classify using a non-linear sequential 
% quadratic programming method (SQP)

% distance, chldren, Idistance, Ichildren, hardClassSize, hardLinkSize,    polyObjectCut1, 2, 3

X0 = [2.7516 6.1991 2.5 3 9 2000 7 ...
     1 1 1 1 1 1 1 1 1 1 ...
     1 1 1 1 1 1 1 1 1 1 ...
     1 1 1 1 1 1 1 1 1 1 ...
     1 1 1 1 1 1 ];

A = [];
b = [];
Aeq = [];
Beq = [];
ub = [10,10,10,10,20,10000,10, ...
      10,10,10,10,10,10,10,10,10,10, ...
      10,10,10,10,10,10,10,10,10,10, ...
      10,10,10,10,10,10,10,10,10,10, ...
      10,10,10,10,10,10 ];
  
lb = [0,0,0,0,0,0,0, ...
      0,0,0,0,0,0,0,0,0,0, ...
      0,0,0,0,0,0,0,0,0,0, ...
      0,0,0,0,0,0,0,0,0,0, ...
      0,0,0,0,0,0,0];
NONLCON = [];

methodType = 'fmincon';
X0
%ub
%lb
time = clock;
cd /home/nathan/aeroru/saliency/matlab
command = ['echo STARTING: METHOD ',methodType,' ',date,' ',...
        num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
        ' >> /home/nathan/aeroru/saliency/matlab/trainNPclassify2Feature.log'];
unix(command);

%path(path,'/lab/mundhenk/code/saliency/matlab');
%options = optimset('DiffMinChange',0.1);

%[x,fval,exitFlag,output,jacobian] = fsolve(@trainNPclassify2Feature,X0,options);
%doThis = inline('trainNPclassifyReal(X0)');
fm = @fmincon;
doThis = @trainNPclassifyReal2;
[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(doThis,X0,A,b,Aeq,Beq,lb,ub,NONLCON);
