%NPclassify_fmincon
%
% This will attempt to minimize the error from NP-classify using a non-linear sequential
% quadratic programming method (SQP)

% distance, chldren, Idistance, Ichildren, hardClassSize, hardLinkSize,    polyObjectCut1, 2, 3

%X0 = [2;7.3381;5.6222;3;5;100];
%X0 = [2; 6; 2.5; 3; 5; 1;   -0.051474;0.48111;-0.19432]; %featrues
%X0 = [2; 6; 2.5; 3; 5; 2]; %images
%X0 = [1.3722 5.9 2.5 3 5 2.8148];
%X0 = [1.8333 5.9524 2.5 3 4.899 2.8148 1]
%X0 = [1.8333 5.9524 2.5 3 10 2.8148 7];
X0 = [2.7516 6.1991 2.5 3 9.899 2.8148 7];
%X0 = [2;6.0297;2.5693;3.5444;5.4355;133.0928;-0.051474;0.48111;0.53811];
%X0 = [2;6.0297;2.955;-8.5864;5.1616;201.8298;-0.051474;0.48111;0.53811];
%X0 = [2;5.949;2.3447;-0.77114;9.1723;27.6191;-0.0024981;1.415;0.74183];
%X0 = [2;5.9417;2.5703;-0.20775;9.6807;29.1816;-0.0098149;1.6478;0.67709];
%A = [[-1,-1,-1,-1,-1,-1];[1,1,1,1,1,1]];

A = [];
b = [];
Aeq = [];
Beq = [];
ub = [10,10,10,10,20,10,10];
lb = [0,0,0,0,0,0,0];
NONLCON = [];
%b = [[0];[500]];

%X0 = [10; 10; 10];
%A = [[-1,-2,-2];[1,2,2]];
%b = [[0];[72]];

methodType = 'fmincon';
X0
%ub
%lb
time = clock;
command = ['echo STARTING: METHOD ',methodType,' ',date,' ',...
        num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
        ' >> /lab/mundhenk/code/saliency/src/trainNPclassify2Feature.log'];
unix(command);

%path(path,'/lab/mundhenk/code/saliency/matlab')
;
%cd /lab/mundhenk/code/saliency/matlab;
options = optimset('DiffMinChange',0.1);

%[x,fval,exitflag,output,lambda] = fmincon(@trainNPclassify2,X0,A,b);
%[x,fval,exitflag,output,lambda] = fmincon(@simpleTest,X0,A,b);
%[x,fval,exitFlag,output,jacobian] = fsolve(@trainNPclassify2Feature,X0,options);
%doThis = inline('trainNPclassifyReal(X0)');
fm = @fmincon;
doThis = @trainNPclassifyReal;
[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(doThis,X0,A,b,Aeq,Beq,lb,ub,NONLCON,options);
%[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(doThis,X0,A,b,Aeq,Beq,lb,ub,NONLCON,options);
%[x,fval,exitFlag,output] = fminsearch(@trainNPclassifyReal,X0,options);