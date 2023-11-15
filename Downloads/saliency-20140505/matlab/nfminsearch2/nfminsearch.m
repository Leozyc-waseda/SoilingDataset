function [x,fval,exitflag,output] = nfminsearch2(funfcn,x,options,varargin)
%NFMINSEARCH Multidimensional unconstrained nonlinear minimization (Nelder-Mead).
%   X = FMINSEARCH(FUN,X0) starts at X0 and finds a local minimizer X of the
%   function FUN. FUN accepts input X and returns a scalar function value 
%   F evaluated at X. X0 can be a scalar, vector or matrix. 
%
%   X = FMINSEARCH(FUN,X0,OPTIONS)  minimizes with the default optimization
%   parameters replaced by values in the structure OPTIONS, created
%   with the OPTIMSET function.  See OPTIMSET for details.  FMINSEARCH uses
%   these options: Display, TolX, TolFun, MaxFunEvals, and MaxIter. 
%
%   X = FMINSEARCH(FUN,X0,OPTIONS,P1,P2,...) provides for additional
%   arguments which are passed to the objective function, F=feval(FUN,X,P1,P2,...).
%   Pass an empty matrix for OPTIONS to use the default values.
%   (Use OPTIONS = [] as a place holder if no options are set.)
%
%   [X,FVAL]= FMINSEARCH(...) returns the value of the objective function,
%   described in FUN, at X.
%
%   [X,FVAL,EXITFLAG] = FMINSEARCH(...) returns a string EXITFLAG that 
%   describes the exit condition of FMINSEARCH.  
%   If EXITFLAG is:
%     1 then FMINSEARCH converged with a solution X.
%     0 then the maximum number of iterations was reached.
%   
%   [X,FVAL,EXITFLAG,OUTPUT] = FMINSEARCH(...) returns a structure
%   OUTPUT with the number of iterations taken in OUTPUT.iterations.
%
%   Examples
%     FUN can be specified using @:
%        X = fminsearch(@sin,3)
%     finds a minimum of the SIN function near 3.
%     In this case, SIN is a function that returns a scalar function value 
%     SIN evaluated at X.
%
%     FUN can also be an inline object:
%        X = fminsearch(inline('norm(x)'),[1;2;3])
%     returns a minimum near [0;0;0].
%
%   FMINSEARCH uses the Nelder-Mead simplex (direct search) method.
%
%   See also OPTIMSET, FMINBND, @, INLINE.

%   Reference: Jeffrey C. Lagarias, James A. Reeds, Margaret H. Wright,
%   Paul E. Wright, "Convergence Properties of the Nelder-Mead Simplex
%   Method in Low Dimensions", SIAM Journal of Optimization, 9(1): 
%   p.112-147, 1998.

%   Copyright 1984-2002 The MathWorks, Inc. 
%   $Revision: 1.21 $  $Date: 2002/04/08 20:26:45 $

%fprintf('\n>>>>>>>RUNNING NATES FMINSEARCH<<<<<<<<\n');
%varargin{4}.ErrorType 

global ITERCOUNT;
ITERCOUNT = 0;

global FUNC_EVALS;
FUNC_EVALS = 0;

global LASTCAT;
LASTCAT = 'none';

global BEST_VAL;
global LOC_BEST_VAL; 
global BESTFILE;
global OUTERLOOPS;
global EOPTIONS;
global RESULTS;
global OUTLIERS;
global ERRORRECORD;
global ERRORFILE;
global DEBUGFILE;
global LOGFILE;

eoptions = EOPTIONS;

defaultopt = struct('Display','notify','MaxIter','200*numberOfVariables',...
   'MaxFunEvals','200*numberOfVariables','TolX',1e-4,'TolFun',1e-4);

% If just 'defaults' passed in, return the default options in X
if nargin==1 & nargout <= 1 & isequal(funfcn,'defaults')
   x = defaultopt;
   return
end

if nargin < 2, 
   error('FMINSEARCH requires at least two input arguments'); 
end

if nargin<3, options = []; end
n                 = prod(size(x));
numberOfVariables = n;

printtype = optimget(options,'Display',defaultopt,'fast');
tolx      = optimget(options,'TolX',defaultopt,'fast');
tolf      = optimget(options,'TolFun',defaultopt,'fast');
maxfun    = optimget(options,'MaxFunEvals',defaultopt,'fast');
maxiter   = optimget(options,'MaxIter',defaultopt,'fast');

% In case the defaults were gathered from calling: optimset('fminsearch'):
if ischar(maxfun)
   if isequal(lower(maxfun),'200*numberofvariables')
      maxfun = 200*numberOfVariables;
   else
      error('Option ''MaxFunEvals'' must be an integer value if not the default.')
   end
end
if ischar(maxiter)
   if isequal(lower(maxiter),'200*numberofvariables')
      maxiter = 200*numberOfVariables;
   else
      error('Option ''MaxIter'' must be an integer value if not the default.')
   end
end

switch printtype
case 'notify'
   prnt = 1;
case {'none','off'}
   prnt = 0;
case 'iter'
   prnt = 3;
case 'final'
   prnt = 2;
case 'simplex'
   prnt = 4;
otherwise
   prnt = 1;
end

header = ' Iteration   Func-count     min f(x)         Procedure';

% Convert to inline function as needed.
funfcn = fcnchk(funfcn,length(varargin));

n = prod(size(x));

% Initialize parameters

%=================================
% NEW SIMPLEX PARAMS - NATE
%=================================
% rho   - Outside expansion / contraction
% chi   - Expansion
% psi   - Contraction
% Sigma - Inside contraction / shrink

%rho = 1; chi = 4; psi = 0.25; sigma = 0.5;
%rho = 0.25; chi = 2; psi = 0.5; sigma = 0.25;
%rho = 4; chi = 2; psi = 0.5; sigma = 0.75;
%rho = 1; chi = 2.5; psi = 0.5; sigma = 0.75;

% If we shrink and keep the same best f this many times then we need to
% intoduce random mutations
if isfield(eoptions,'Max_bsCount')
    max_bsCount = eoptions.Max_bsCount;
else
    max_bsCount = 2;
end
% should we keep the datasets orthogonal?
if isfield(eoptions,'BasicOrtho')
    basicOrtho = eoptions.BasicOrtho;
else
    basicOrtho = 0;
end

if isfield(eoptions,'Rho')
    rho = eoptions.Rho;
else
    rho = 1; 
end

if isfield(eoptions,'Chi')
    chi = eoptions.Chi;
else
    chi = 2;
end

if isfield(eoptions,'Psi')
    psi = eoptions.Psi;
else
    psi = 0.5;
end

if isfield(eoptions,'Sigma')
    sigma = eoptions.Sigma;
else
    sigma = 0.5;
end

if isfield(eoptions,'Theta')
    theta = eoptions.Theta;
else
    theta = 0.5;
end

if isfield(eoptions,'ErrorType')
    errorType = eoptions.ErrorType;
else
    errorType = 'Unspecified!';
end

%=================================
% Reflection          - xr     = (1 + rho)*xbar     - rho*v(:,end);
% Expansion           - xe     = (1 + rho*chi)*xbar - rho*chi*v(:,end);
% Outside Contraction - xc     = (1 + rho*psi)*xbar - rho*psi*v(:,end);
% Inside Contraction  - xcc    = (1 - psi)*xbar     + psi*v(:,end);
% Shrink              - v(:,j) = v(:,1)+sigma*(v(:,j) - v(:,1));

%=================================
% NEW SIMPLEX PARAMS - NATE
%=================================
%zero_term_delta  = 0.025;
%usual_delta      = 0.25;

% 5 percent deltas for non-zero terms
if isfield(eoptions,'UsualDelta')
    usual_delta = eoptions.UsualDelta;
else
    usual_delta = 0.05;
end

% Even smaller delta for zero elements of x
if isfield(eoptions,'ZeroTermDelta')
    zero_term_delta = eoptions.ZeroTermDelta;
else
    zero_term_delta = 0.00025;      
end

%=================================
% useRandom 
% 0 - Orignal Simplex
% 1 - Randomize the jth variable in n 
% 2 - Randomize x% of each n
% 3 - Randomize all
if isfield(eoptions,'UseRandom')
    useRandom = eoptions.UseRandom;
else
    useRandom = 2;
end
% Spread of random varaibles about 0, 2 would be -1 to 1
if isfield(eoptions,'RandSpread')
    randSpread = eoptions.RandSpread;
else
    randSpread = 2;
end
% what is the initial value of bestShrink
if isfield(eoptions,'BestShrink')
    bestShrink = eoptions.BestShrink;
else
    bestShrink = 0;
end
% what percent of orignal data should be kept on average durring useRandom = 2
if isfield(eoptions,'RandKeep')
    randKeep = eoptions.RandKeep;
else
    randKeep = 0.75;
end
% Durring cataclism, if we mutate, what percent of data to keep
if isfield(eoptions,'RandKeepMut')
    randKeepMut = eoptions.RandKeepMut;
else
    randKeepMut = 0.75;
end
% Durring death cataclism, what percent to kill off
if isfield(eoptions,'RandKeepCat')
    randKeepCat = eoptions.RandKeepCat;
else
    randKeepCat = 0.5;
end
% Odds favoring a mutation caticlism over a straight up death cataclism
if isfield(eoptions,'MutOdds')
    mutOdds = eoptions.MutOdds;
else
    mutOdds = 0.5;
end
% Should we allow cataclisms? 1 = yes, 0 = no
if isfield(eoptions,'UseCat')
    useCat = eoptions.UseCat;
else
    useCat = 1;
end

if ~isfield(eoptions,'UseStoch')
    eoptions.UseStoch = 'yes';
end

onesn   = ones(1,n);
two2np1 = 2:n+1;
one2n   = 1:n;
fullN   = n+1;

% Set up a simplex near the initial guess.

xin = x(:);       % Force xin to be a column vector

v = zeros(n,n+1); fv = zeros(1,n+1);

if useRandom == 3 
    y = xin;
    for i = 1:n
        if isfield(eoptions,'ControlRand') & strcmp(eoptions.ControlRand,'yes')
            new_randSpread = randSpread(:,i);
        else
            new_randSpread = randSpread;
        end
        y(i) = (rand(1,1) - 0.5) * new_randSpread;
    end
    y = nfminclamp(y);
    if basicOrtho == 1
        y = call_normalize_pos_neg(y);
    end
    v(:,1)  = y;      % Place random guess in the simplex!
    x(:)    = y;      % Change x to the form expected by funfcn 
else
    xin = nfminclamp(xin);
    if basicOrtho == 1
        xin = call_normalize_pos_neg(xin);
    end
    v(:,1)  = xin;    % Place input guess in the simplex! (credit L.Pfeffer at Stanford)
    x(:)    = xin;    % Change x to the form expected by funfcn 
end

fv(:,1) = feval(funfcn,x,varargin{:}); 
LOC_BEST_VAL = fv(:,1);

for j = 1:n
    y = xin;
    if useRandom == 0
        if y(j) ~= 0
            y(j) = (1 + usual_delta)*y(j);
        else
            y(j) = zero_term_delta;
        end 
    elseif useRandom == 1
        if isfield(eoptions,'ControlRand') & strcmp(eoptions.ControlRand,'yes')
            new_randSpread = randSpread(:,i);
        else
            new_randSpread = randSpread;
        end
        y(j) = (rand(1,1) - 0.5) * new_randSpread;
    elseif useRandom == 2
        for i = 1:n
            doRand = rand(1,1);
            if doRand > randKeep
                if isfield(eoptions,'ControlRand') & strcmp(eoptions.ControlRand,'yes')
                    new_randSpread = randSpread(:,i);
                else
                    new_randSpread = randSpread;
                end
                y(i) = (rand(1,1) - 0.5) * new_randSpread;
            end
        end
    elseif useRandom == 3
        for i = 1:n
            if isfield(eoptions,'ControlRand') & strcmp(eoptions.ControlRand,'yes')
                new_randSpread = randSpread(:,i);
            else
                new_randSpread = randSpread;
            end
            y(i) = (rand(1,1) - 0.5) * new_randSpread;
        end
    end
    y = nfminclamp(y);
    if basicOrtho == 1
        y = call_normalize_pos_neg(y);    
    end
    
   v(:,j+1) = y;
   %fid = fopen('NN_logfile_ext.txt','a');
   %fprintf(fid,'SIMPLEX - Start Up %d\n',j);
   %fclose(fid);  
   x(:) = y; f = feval(funfcn,x,varargin{:});
   fv(1,j+1) = f; 
   fid = fopen(LOGFILE,'a');
   fprintf(fid,'SIMPLEX - Start Up %d ERROR: %f\n',j,f);
   fclose(fid);
end     

% sort so v(1,:) has the lowest function value 
[fv,j] = sort(fv);
v = v(:,j);

how = 'initial';

ITERCOUNT  = 1;
FUNC_EVALS = n+1
errorRecord      = struct('Description','Holds all the errors for all operations per iteration');
errorRecord.data = struct('Description','Holds the actual simplex parameters');
errorRecord.time = struct('Description','Holds the current time stamp of storage');
lstats           = struct('Description','local stats');

if prnt == 3
   disp(' ')
   disp(header)
   disp([sprintf(' %5.0f        %5.0f     %12.6g         ', ITERCOUNT, FUNC_EVALS, fv(1)), how]) 
elseif prnt == 4
   clc
   formatsave = get(0,{'format','formatspacing'});
   format compact
   format short e
   disp(' ')
   disp(how)
   v
   fv
   FUNC_EVALS
end
exitflag = 1;
bsCount = 0;
OUTLIERS.Hold = 0;
% Main algorithm
% Iterate until the diameter of the simplex is less than tolx
%   AND the function values differ from the min by less than tolf,
%   or the max function evaluations are exceeded. (Cannot use OR instead of AND.)
while FUNC_EVALS < maxfun & ITERCOUNT < maxiter
   if max(max(abs(v(:,two2np1)-v(:,onesn)))) <= tolx & ...
         max(abs(fv(1)-fv(two2np1))) <= tolf
      break
   end
   how = '';
  
   shrink = 0;
   % Compute the reflection point
   
   % xbar = average of the n (NOT n+1) best points
   xbar = sum(v(:,one2n), 2)/n;
   
   %%%%%%
   % Compute Difference between mean and worst result ->
   xr = nfminclamp((1 + rho)*xbar - rho*v(:,end));
   if basicOrtho == 1
       xr = call_normalize_pos_neg(xr);
   end
   if eoptions.debugLevel > 0
        fid = fopen(LOGFILE,'a');
        fprintf(fid,'SIMPLEX - Reflexion Point\n');
        fclose(fid);
   end
   x(:) = xr; fxr = feval(funfcn,x,varargin{:});
   errorRecord.reflect      = fxr;
   errorRecord.data.reflect = xr;
   errorRecord.time.reflect = ITERCOUNT;
   FUNC_EVALS = FUNC_EVALS+1;
   
   %%%%%%
   % if the reflection is better than the best result ->
   if fxr < fv(:,1) & EOPTIONS.ForceCataclism == 0
      % Calculate the expansion point
      xe = nfminclamp((1 + rho*chi)*xbar - rho*chi*v(:,end));
      if basicOrtho == 1
          xe = call_normalize_pos_neg(xe);
      end
      if eoptions.debugLevel > 0
          fid = fopen(LOGFILE,'a');
          fprintf(fid,'SIMPLEX - Expansion Point\n');
          fclose(fid);
      end
      x(:) = xe; fxe = feval(funfcn,x,varargin{:});
      errorRecord.expand      = fxe;
      errorRecord.data.expand = xe;
      errorRecord.time.expand = ITERCOUNT;
      FUNC_EVALS = FUNC_EVALS+1;
      if fxe < fxr
         v(:,end) = xe;     % <- Replace the worst result
         fv(:,end) = fxe;   % <- Replace the worst result
         how = 'expand';
      else
         v(:,end) = xr;     % <- Replace the worst result
         fv(:,end) = fxr;   % <- Replace the worst result
         how = 'reflect';
      end
   %%%%%%   
   % The Reflection is not better than the best result ->   
   else % fv(:,1) <= fxr
      if fxr < fv(:,n) & EOPTIONS.ForceCataclism == 0
         v(:,end) = xr; 
         fv(:,end) = fxr;
         how = 'reflect';
      else % fxr >= fv(:,n) 
         % Perform contraction
         if fxr < fv(:,end) & EOPTIONS.ForceCataclism == 0
            % Perform an outside contraction
            xc = nfminclamp((1 + psi*rho)*xbar - psi*rho*v(:,end));

            if basicOrtho == 1
                xc = call_normalize_pos_neg(xc);
            end
            if eoptions.debugLevel > 0
                fid = fopen(LOGFILE,'a');
                fprintf(fid,'SIMPLEX - Outside Contraction\n');
                fclose(fid);
            end
            x(:) = xc; fxc = feval(funfcn,x,varargin{:});
            errorRecord.contract_outside      = fxc;
            errorRecord.data.contract_outside = xc;
            errorRecord.time.contract_outside = ITERCOUNT;
            FUNC_EVALS = FUNC_EVALS+1;
            
            if fxc <= fxr
               v(:,end) = xc; 
               fv(:,end) = fxc;
               how = 'contract_outside';
            else
               % perform a shrink
               how = 'shrink'; 
            end
        elseif EOPTIONS.ForceCataclism == 0
            % Perform an inside contraction
            xcc = nfminclamp((1-psi)*xbar + psi*v(:,end));
            
            if basicOrtho == 1
                xcc = call_normalize_pos_neg(xcc);
            end
            if eoptions.debugLevel > 0
                fid = fopen(LOGFILE,'a');
                fprintf(fid,'SIMPLEX - Inside Contraction\n');
                fclose(fid);
            end
            x(:) = xcc; fxcc = feval(funfcn,x,varargin{:});
            errorRecord.contract_inside      = fxcc;
            errorRecord.data.contract_inside = xcc;
            errorRecord.time.contract_inside = ITERCOUNT;
            FUNC_EVALS = FUNC_EVALS+1;
            
            if fxcc < fv(:,end)
               v(:,end) = xcc;
               fv(:,end) = fxcc;
               how = 'contract_inside';
            else
               % perform a shrink
               how = 'shrink';
            end
         end
         if strcmp(how,'shrink') & EOPTIONS.ForceCataclism == 0
            if strcmp(eoptions.UseStoch,'yes')
              % Try a stochastic jump - NATE
                xs = nfminclamp(((max(v(:,end)) - min(v(:,end))) * theta * ((rand(size(v(:,end),1),1) - 0.5) * 2)) + v(:,end));
                
                if basicOrtho == 1
                    xs = call_normalize_pos_neg(xs);
                end
                if eoptions.debugLevel > 0
                    fid = fopen(LOGFILE,'a');
                    fprintf(fid,'SIMPLEX - Stochastic min-max\n');
                    fclose(fid);
                end
                x(:) = xs; fxs = feval(funfcn,x,varargin{:});
                errorRecord.stochastic_min_max      = fxs;
                errorRecord.data.stochastic_min_max = xs;
                errorRecord.time.stochastic_min_max = ITERCOUNT;
                FUNC_EVALS = FUNC_EVALS+1;
                if fxs < fv(:,end) 
                    v(:,end) = xs;
                    fv(:,end) = fxs;
                    how = 'stochastic_min_max';
                else

                    xss = nfminclamp(((max(v(:,end) - xbar)) * theta * ((rand(size(v(:,end),1),1) - 0.5) * 2)) + v(:,end));
                    
                    if basicOrtho == 1
                        xss = call_normalize_pos_neg(xss);
                    end
                    if eoptions.debugLevel > 0
                        fid = fopen(LOGFILE,'a');
                        fprintf(fid,'SIMPLEX - Stochastic max-xbar\n');
                        fclose(fid);
                    end
                    x(:) = xss; fxss = feval(funfcn,x,varargin{:});
                    errorRecord.stochastic_max_xbar      = fxss;
                    errorRecord.data.stochastic_max_xbar = xss;
                    errorRecord.time.stochastic_max_xbar = ITERCOUNT;
                    FUNC_EVALS = FUNC_EVALS+1;
                    if fxss < fv(:,end)
                        v(:,end) = xss;
                        fv(:,end) = fxss;
                        how = 'stochastic_max_xbar';
                    else

                        xsa = nfminclamp(((min(v(:,end) - xbar)) * theta * ((rand(size(v(:,end),1),1) - 0.5) * 2)) + v(:,end));
                       
                        if basicOrtho == 1
                            xsa = call_normalize_pos_neg(xsa);
                        end
                        if eoptions.debugLevel > 0
                            fid = fopen(LOGFILE,'a');
                            fprintf(fid,'SIMPLEX - Stochastic min-xbar\n');
                            fclose(fid);
                        end
                        x(:) = xsa; fxsa = feval(funfcn,x,varargin{:});
                        errorRecord.stochastic_min_xbar      = fxsa;
                        errorRecord.data.stochastic_min_xbar = xsa;
                        errorRecord.time.stochastic_min_xbar = ITERCOUNT;
                        FUNC_EVALS = FUNC_EVALS+1;
                        if fxsa < fv(:,end) 
                            v(:,end) = xsa;
                            fv(:,end) = fxsa;
                            how = 'stochastic_min_xbar';
                        else
                            how = 'shrink';
                            shrink = 1;
                        end
                    end
                end
            end 
        end
        if strcmp(how,'shrink') & EOPTIONS.ForceCataclism == 0  
            errorRecord.shrink{1}      = fv(:,1);
            errorRecord.data.shrink{1} = v(:,1);
            errorRecord.time.shrink{1} = ITERCOUNT;
            for j=two2np1
                if EOPTIONS.ForceCataclism == 0

                    v(:,j) = nfminclamp(v(:,1)+sigma*(v(:,j) - v(:,1)));
                    
                    if basicOrtho == 1
                        v(:,j) = call_normalize_pos_neg(v(:,j));
                    end
                    x(:) = v(:,j); 
                    fv(:,j) = feval(funfcn,x,varargin{:});
                    errorRecord.shrink{j}      = fv(:,j);
                    errorRecord.data.shrink{j} = v(:,j);
                    errorRecord.time.shrink{j} = ITERCOUNT;
                    if eoptions.debugLevel > 0
                        fid = fopen(LOGFILE,'a');
                        fprintf(fid,'SIMPLEX - Shrink Norm %d\n',j);
                        fclose(fid);
                    end
                    %x(:) = vs;  fvs = feval(funfcn,x,varargin{:});
                    %v(:,j) = vs; 
                    %fv(:,j) = fvs;
                end
            end
         end
         FUNC_EVALS = FUNC_EVALS + n;
      end
   end
   [fv,j] = sort(fv);
   v = v(:,j);
   if fv(:,1) < LOC_BEST_VAL
       LOC_BEST_VAL = fv(:,1);
   end
   if fv(:,1) < BEST_VAL 
       RESULTS.Simplex.BestParamSet = v(:,1);
       RESULTS.Simplex.BestParamVal = fv(:,1);
       BEST_VAL = fv(:,1);
       ERRORRECORD = errorRecord;
       
       fid = fopen(BESTFILE,'a');
       if isfield(eoptions,'UseProjectionPersuit') & strcmp(eoptions.UseProjectionPersuit,'yes')
            [eoptions.Soptions,RESULTS.XO] = getProjParamSet('stitch',eoptions.Soptions,v(:,1)');   
            pout = RESULTS.XO';
            fprintf(fid,'PROJDIM %d ',eoptions.Soptions.Proj.WorkDim); 
       else
            RESULTS.XO = v(:,1)';  
            pout       = v(:,1);    
       end
           
       fprintf(fid,'VALUE %f LOCAL %f ERRTYPE %s ITER %d LOOP %d HOW %s LASTCAT %s\n',BEST_VAL,LOC_BEST_VAL,errorType,ITERCOUNT,OUTERLOOPS,how,LASTCAT);
       if strcmp(how,'shrink')
           fprintf(fid,'%18.18f ErrorRecord',BEST_VAL);
           for j=1:fullN
                 fprintf(fid,' %f ',errorRecord.shrink{j});
           end
           fprintf(fid,'\n');
       else
           fprintf(fid,'%18.18f\n',BEST_VAL);
       end
       
       for i = 1 : size(pout,1)
            fprintf(fid,'%64.64f\t',pout(i,1));
       end
       fprintf(fid,'\n');
       fclose(fid);
          
       if eoptions.debugLevel > 0
            lstats.BEST_VAL     = BEST_VAL;
            lstats.LOC_BEST_VAL = LOC_BEST_VAL;
            lstats.errorType    = errorType;
            lstats.ITERCOUNT    = ITERCOUNT;
            lstats.OUTERLOOPS   = OUTERLOOPS;
            lstats.how          = how;
            lstats.LASTCAT      = LASTCAT;
            lstats.fullN      = fullN;
            dumpErrorRecord(ERRORRECORD,ERRORFILE,DEBUGFILE,lstats,eoptions.debugLevel);
       end
   end
   % keep track of lack of progress
   if shrink == 1
        if fv(:,1) == bestShrink
            bsCount = bsCount + 1;
        else
            bestShrink = fv(:,1);
            bsCount = 0;
        end
    end
   
    % introduce a cataclism of random mutations or extinction (solar flairs
    % v. giant meteor)
    if ( bsCount == max_bsCount & useCat == 1 ) | EOPTIONS.ForceCataclism == 1
        %fid = fopen(LOGFILE,'a');
        %fprintf(fid,'>>>SIMPLEX - CATACLISM<<<\n');
        %fclose(fid);
        OUTLIERS.Hold = 1;
        catRand = rand(1,1);
        catType = '';
        
        if EOPTIONS.ForceCataclism == 1
            temp                     = varargin{4}.ErrorType;
            varargin{4}.ErrorType    = varargin{4}.ErrorTypeAlt;
            varargin{4}.ErrorTypeAlt = temp;
        end
        
        if catRand <= mutOdds | EOPTIONS.ForceCataclism == 1  
            if EOPTIONS.ForceCataclism == 1
                catType = 'force';
            else
                catType = 'mutate';
            end
            LASTCAT = catType;
            for i = 1 : size(v,1)
                for j = 1 : size(v,2)
                    doRand = rand(1,1);
                    if doRand > randKeepMut
                        if isfield(eoptions,'ControlRand') & strcmp(eoptions.ControlRand,'yes')
                            new_randSpread = randSpread(:,i);
                        else
                            new_randSpread = randSpread;
                        end
                        v(i,j) = (rand(1,1) - 0.5) * new_randSpread;
                    end
                end
            end
        else
            catType = 'death';
            LASTCAT = catType;
            for j = 1 : size(v,2)
                doRand = rand(1,1);
                if doRand > randKeepCat
                    for i = 1 : size(v,1)
                        if isfield(eoptions,'ControlRand') & strcmp(eoptions.ControlRand,'yes')
                            new_randSpread = randSpread(:,i);
                        else
                            new_randSpread = randSpread;
                        end
                        v(i,j) = (rand(1,1) - 0.5) * new_randSpread;
                    end
                end
            end
        end
        fprintf('>>>CATACLISM of type %s\n',LASTCAT);
        
        % re-evaluate the whole thing
        for j = 1 : size(v,2)
            vm = nfminclamp(v(:,j));
            if basicOrtho == 1
                vm = call_normalize_pos_neg(vm);
            end
            if eoptions.debugLevel > 0
                fid = fopen(LOGFILE,'a');
                fprintf(fid,'SIMPLEX - cataclism %s %d\n',catType,j);
                fclose(fid);
            end
            x(:) = vm; fm = feval(funfcn,x,varargin{:});
            errorRecord.mutate{j}      = fm;
            errorRecord.data.mutate{j} = vm;
            errorRecord.time.mutate{j} = ITERCOUNT;
            v(:,j) = vm; 
            fv(:,j) = fm;
        end
        
        % re-sort
        [fv,j] = sort(fv);
        v = v(:,j);
        
        % reset counter
        bsCount = 0;
        how = 'mutate'; 
        if fv(:,1) < LOC_BEST_VAL
            LOC_BEST_VAL = fv(:,1);
        end
        if fv(:,1) < BEST_VAL | EOPTIONS.ForceCataclism == 1;  
            fprintf('>>>CATACLISM new best found %f\n',fv(:,1));
            RESULTS.Simplex.BestParamSet = v(:,1);
            RESULTS.Simplex.BestParamVal = fv(:,1);
            BEST_VAL = fv(:,1);
            ERRORRECORD = errorRecord;
            
            fid = fopen(BESTFILE,'a');
            if isfield(eoptions,'UseProjectionPersuit') & strcmp(eoptions.UseProjectionPersuit,'yes')
                [eoptions.Soptions,RESULTS.XO] = getProjParamSet('stitch',eoptions.Soptions,v(:,1)');   
                pout = RESULTS.XO';
                fprintf(fid,'PROJDIM %d ',eoptions.Soptions.Proj.WorkDim); 
            else
                RESULTS.XO = v(:,1)';  
                pout       = v(:,1);    
            end
           
            fprintf(fid,'(NEW CAT) VALUE %f LOCAL %f ERRTYPE %s ITER %d LOOP %d HOW %s LASTCAT %s\n',BEST_VAL,LOC_BEST_VAL,errorType,ITERCOUNT,OUTERLOOPS,how,LASTCAT);
            fprintf(fid,'%18.18f\n',BEST_VAL);
            for i = 1 : size(pout,1)
                fprintf(fid,'%64.64f\t',pout(i,1));
            end
            fprintf(fid,'\n');
            fclose(fid);
            
            if eoptions.debugLevel > 0
                lstats.BEST_VAL     = BEST_VAL;
                lstats.LOC_BEST_VAL = LOC_BEST_VAL;
                lstats.errorType    = errorType;
                lstats.ITERCOUNT    = ITERCOUNT;
                lstats.OUTERLOOPS   = OUTERLOOPS;
                lstats.how          = how;
                lstats.LASTCAT      = LASTCAT;
                lstats.fullN        = fullN;
                dumpErrorRecord(ERRORRECORD,ERRORFILE,DEBUGFILE,lstats,eoptions.debugLevel);
            end
        end
    end
   OUTLIERS.Hold = 0;
   EOPTIONS.ForceCataclism = 0;  
   ITERCOUNT = ITERCOUNT + 1;
   % print to file - NATE
   if eoptions.debugLevel > 0
        fid = fopen(LOGFILE,'a');
        fprintf(fid,'\n>>>>> SIMPLEX Iter %5.0f FUNC_EVALS %5.0f Func_val %12.6g How %s<<<<<\n>>>>>BEST %f <<<<<\n', ITERCOUNT, FUNC_EVALS, fv(1), how, LOC_BEST_VAL);
        fclose(fid');
   end
   
   if prnt == 3
   disp([sprintf(' %5.0f        %5.0f     %12.6g         ', ITERCOUNT, FUNC_EVALS, fv(1)), how]) 
   elseif prnt == 4
      disp(' ')
      disp(how)
      v
      fv
      FUNC_EVALS
   end  
end   % while


x(:) = v(:,1);
if prnt == 4,
   % reset format
   set(0,{'format','formatspacing'},formatsave);
end
output.iterations = ITERCOUNT;
output.funcCount = FUNC_EVALS;
output.algorithm = 'Nelder-Mead simplex direct search';

fval = min(fv); 
if FUNC_EVALS >= maxfun 
   if prnt > 0
      disp(' ')
      disp('Exiting: Maximum number of function evaluations has been exceeded')
      disp('         - increase MaxFunEvals option.')
      msg = sprintf('         Current function value: %f \n', fval);
      disp(msg)
   end
   exitflag = 0;
elseif ITERCOUNT >= maxiter 
   if prnt > 0
      disp(' ')
      disp('Exiting: Maximum number of iterations has been exceeded')
      disp('         - increase MaxIter option.')
      msg = sprintf('         Current function value: %f \n', fval);
      disp(msg)
   end
   exitflag = 0; 
else
   if prnt > 1
      convmsg1 = sprintf([ ...
         '\nOptimization terminated successfully:\n',...
         ' the current x satisfies the termination criteria using OPTIONS.TolX of %e \n',...
         ' and F(X) satisfies the convergence criteria using OPTIONS.TolFun of %e \n'
          ],tolx, tolf);
      disp(convmsg1)
   end
   exitflag = 1;
end

%===================================================================================================
%===================================================================================================
%===================================================================================================

function X = nfminclamp(X)

global EOPTIONS;

if strcmp(EOPTIONS.useUpperBound,'yes')
    for i = 1:size(X,1)
        if X(i,:) > EOPTIONS.upperBound(i,:);
            X(i,:) = EOPTIONS.upperBound(i,:);
        end
    end
end

if strcmp(EOPTIONS.useLowerBound,'yes')
    for i = 1:size(X,1)
        if X(i,:) < EOPTIONS.lowerBound(i,:);
            X(i,:) = EOPTIONS.lowerBound(i,:);
        end
    end
end




