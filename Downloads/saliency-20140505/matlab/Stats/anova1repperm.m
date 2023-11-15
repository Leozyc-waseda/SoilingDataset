function [h,p] = anova1repperm(X,alpha,iterations,statwithin,statbetween)
%A single factor repeated measures significance test to determine if
%factors (columns, repeated measure) of X significantly differ. The
%repeated measures design restricts the permutation to
%within-subject for the repeated factor. The default test statistic
%is to compute sum of squares of column sums. This is an equivalant
%statistic (will yield same p-value) as a 'F statistic' from a
%'1-way ANOVA' for equal means). Alpha is the desired significance
%level(default 0.05). 'iterations' is the number of monte-carlo
%simulations to use (default 30,000). One should perform as many
%simulations as necessary to stabalize the p-value for your desired
%alpha. The user can optionally supply different test statistics for
%within levels (within a single column) and between levels.
%
%references:
%From Edington, ES. (1995). Randomization tests. New York, NY:Marcel
%Dekker inc.
%
%written by David J. Berg (dberg@usc.edu) 

if (nargin < 5)
  statbetween=@(x, N)(sum( (x.^2) ./ N ));
end
if (nargin < 4)
  statwithin=@(X)(sum(X));
end
if (nargin < 3)
  iterations = 30000;
end
if (nargin < 2)
  alpha = 0.05;
end

rows = size(X,1);cols = size(X,2);
N = ones(1,cols);

%compute our first test stat
T = statwithin(X);
T = statbetween(T, N);
count = 1;

xt = zeros(size(X));
for (ii = 2:iterations)%start at 2 because our original counts as 1 permutation
  %compute test stat
  for (jj = 1:rows)
    xt(jj,:) = X(jj,randperm(cols));
  end
  t = statwithin(xt);
  t = statbetween(t, N);
  
  %test and increment counter
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

    