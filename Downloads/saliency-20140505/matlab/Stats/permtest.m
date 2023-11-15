function [h, p] = permtest(x, y, alpha, tail, iterations, teststat)
%Computes a permutation test to determine whether x and y are
%signifantly different using the supplied test statistic. If 'tail'
%is set to 'both' then the sum of squared group sums is used as the
%test statistic and it is an equivelant statitstic (will yield same
%p-value) to a two group 'F' statistic which is equivelant to a
%'t' statistic. If performing a one-tailed test then the sum of
%group1 is used as the test statistic and it is an equivalant
%statistic to a 't' statistic from a '1-tailed t-test'. Alpha is
%the desired significance level(default 0.05). 'tail' can be 'both'
%(default, means are not equal), 'right' (mean of x is greater than
%mean of y), 'left' (mean of x less than mean of y). 'iterations' is
%the number of monte-carlo simulations to use (default 30,000). One
%should perform as many simulations as necessary to stabalize the
%p-value for your desired alpha.
%
%example which should give same result as standart 't-test' (with
%enough iterations):
%x = randn(1,1000)+.1;y = randn(1,1000);
%[h,p] = permtest(x,y)
%[h,p] = ttest2(x,y)
%
%references:
%From Edington, ES. (1995). Randomization tests. New York, NY:Marcel
%Dekker inc.
%
%written by David J. Berg (dberg@usc.edu) 

if (nargin < 5)
  iterations = 30000;
end
if (nargin < 4)
  tail = 'both';
end
if (nargin < 6)
  if (strcmpi(tail,'both'))
    teststat=@(x,y)( (sum(x).^2)./length(x) + (sum(y).^2)./length(y) );
  else
    teststat=@(x,y)(sum(x));
  end
end
if (nargin < 3)
  alpha = 0.05;
end

if (strcmpi(tail,'left'))
  tmp = y;
  y=x;
  x=tmp;
end
  
lx = length(x);
d = [x(:);y(:)]; ld = length(d);

%generate original test. 
T = teststat(x,y);

count = 1;
for (ii = 2:iterations)%start at 2 because our original counts as 1 permutation
  p = randperm(ld);
  xt = d(p(1:lx));
  yt = d(p(lx+1:end));
  t = teststat(xt,yt);
  
  if (t >= T)
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

    