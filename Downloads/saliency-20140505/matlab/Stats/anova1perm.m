function [h,p] = anova1perm(X,group,alpha,iterations,statwithin, statbetween)
%Computes a single factor permutation test to determine whether the
%columns of X significantly differ from one another, using the
%supplied test statistics (default is to compute sum of squares of row
%sums and this is an equivalant statistic (will yield same p-value) to
%a 'F statistic' from a '1-way ANOVA' for equal means). X can also be
%a vector (for unbalanced designs, and if so, group should be a
%numeric vector indicating group assigment for each data point). For
%balanced designs the matrix version will be significantly
%faster. Alpha is the desired significance level(default
%0.05). 'iterations' is the number of monte-carlo simulations to use
%(default 30,000). One should perform as many simulations as necessary
%to stabalize the p-value for your desired alpha. The user can
%optionally supply different test statistics for within levels (in
%a single column) and between levels. 
%
%example which should give same result as standart 'ANOVA' (with
%enough iterations):
%X = randn(1000,3);X(:,2) = X(:,2) + .15 %three groups each with 100 samples
%[h,p] = anova1perm(X)
%[h,p] = anova1(X)
%
%references:
%From Edington, ES. (1995). Randomization tests. New York, NY:Marcel
%Dekker inc.
%
%written by David J. Berg (dberg@usc.edu) 

if (nargin < 6)
  statbetween=@(x, N)(sum( (x.^2) ./ N ));
end
if (nargin < 5)
  statwithin=@(X)(sum(X));
end
if (nargin < 4)
  iterations = 30000;
end
if (nargin < 3)
  alpha = 0.05;
end
if (nargin < 2)
  if (isvector(X))
    error('if group is not supplied, X must matrix');
  end
end

rows = size(X,1);cols = size(X,2);
N = ones(1,cols);

%compute index for each group
if (isvector(X))
  g = unique(group);
  for (ii = 1:length(g))
    index(ii) = length(find(group == g(ii)));
  end
  N = index;
  index = [0,index];
  index = cumsum(index);
end

%if X is already a vector, this wont do anything
d = X(:); ld = length(d);

%compute our first test stat
if (~isvector(X))
  T = statwithin(X);
  T = statbetween(T, N);
else
  for (ii = 1:length(g))
    f = find(group == g(ii));
    T(ii) = statwithin(d(f));
  end
  T = statbetween(T, N);
end

count = 1;
for (ii = 2:iterations)%start at 2 because our original counts as 1 permutation
  p = randperm(ld);
  t = [];  
  
  %compute test stat
  if (~isvector(X))
    xt = d(p);
    xt = reshape(xt,rows,cols);
    t = statwithin(xt);
    t = statbetween(t, N);
  else
    for (ii = 1:length(g))
      t(ii) = statwithin(d(p(index(ii)+1:index(ii+1))));
    end
    t = statbetween(t, N);
  end

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

    