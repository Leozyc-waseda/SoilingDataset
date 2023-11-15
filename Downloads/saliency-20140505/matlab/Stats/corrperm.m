function [h, p, stat] = corrperm(x,y,repeat,alpha,iterations,errfunc)
%This is a general purpose tool for computing significance of
%correlation or goodness of fit between two vectors or matricies. If x
%and y are matricies then it is assumed that rows represent
%different subjects. Matricies and vector must be of equal size. 'repeat'
%should be set to 1 if the columns are repeated measures (restricts
%permutation). Alpha is the desired significance level (default
%0.05). 'iterations' is the number of monte-carlo simulations to use
%(default 30,000). One should perform as many simulations as necessary
%to stabalize the p-value for your desired alpha. 'errfunc' can be any
%two tailed error function. Several are given or the user can pass
%their own function handle. The error function should work along all
%elements of the matrix and return a single test statistic. Use 'corr'
%for the the sum(x.*y) which is an equivalant statistic (will yield
%same p-value) as a Pearson correlation coefficient to test for linear
%correlation. Use 'sumsquare' for sum of squared error. Use
%'sumabsdiff' for sum of absolute difference.
%
%references:
%From Edington, ES. (1995). Randomization tests. New York, NY:Marcel
%Dekker inc.
%
%written by David J. Berg (dberg@usc.edu)
if (nargin < 6)
  errfunc = @(x,y)(-1*abs( length(x(:))*(x(:)'*y(:)) - sum(x(:))*sum(y(:)) ));
else
  if (isstr(errfunc))
    if (strcmpi(errfunc,'corr'))
      errfunc = @(x,y)(-1*abs(length(x(:))*(x(:)'*y(:))-sum(x(:))*sum(y(:))));
    elseif (strcmpi(errfunc,'sumsquare'))
      errfunc = @(x,y)sum((x(:)-y(:)).^2);
    elseif(strcmpi(errfunc,'sumabsdiff'))
      errfunc = @(x,y)sum(abs(x(:)-y(:)));
    end
  end
end
if (nargin < 5)
  iterations = 30000;
end
if (nargin < 4)
  alpha = 0.05;
end
if (nargin < 3)
  repeat = 0;
end

if ((size(x,1) ~= size(y,1)) || (size(x,2) ~= size(y,2)))
  error('x and y must be the same number of columns');
end

%compute our first test stat
T = errfunc(x, y);
count = 1;

yt = zeros(size(y));
L = length(y(:));
rows = size(yt,1);cols = size(x,2);

for (ii = 2:iterations)%start at 2 because our original counts as 1 permutation
  %compute test stat
  if (repeat)
    for (jj = 1:rows)
      yt(jj,:) = y(jj,randperm(cols));
    end
  else
    yt = reshape(y(randperm(L)),[],cols);
  end
  
  t = errfunc(x, yt);
  %test and increment counter
  if (t <= T)
    count = count + 1;
  end
end

%compute p and return if significant
p = count / iterations;
if (p < alpha)
  h = 1;
else 
  h = 0;
end
stat = T;

    