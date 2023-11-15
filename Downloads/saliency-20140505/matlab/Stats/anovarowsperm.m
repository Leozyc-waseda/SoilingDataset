function [h,p] = anovarowsperm(X, group, alpha, iterations, ...
                               statwithinsub, statbetweengroups)
%Computes a main effect by permuting entire rows as a single unit. X
%must be a matrix where each row is a subject, and each col is a
%measurement (possibly a repeated factor, multiple measured variables
%etc). Group should be a vector of integer values indicating to which
%group each row of X belongs. The default test statitstic is to
%compute the sum within subjects (across colums), then sum within a
%group, then square and sum between groups. This is an equivalant
%statistic (will yield same p-value) as a 'F statistic.' Alpha is the
%desired significance level (default 0.05). 'iterations' is the number
%of monte-carlo simulations to use (default 30,000). The user can
%optionally supply different test statistics for within subject (a
%single row) and between groups.
%
%references:
%From Edington, ES. (1995). Randomization tests. New York, NY:Marcel
%Dekker inc.
%
%written by David J. Berg (dberg@usc.edu) 

if (nargin < 6)
  statbetweengroups=@(x, N)( sum((x.^2) ./ N) );
end
if (nargin < 5)
  statwithinsub=@(X)(sum(X,2));
end
if (nargin < 4)
  iterations = 30000;
end
if (nargin < 3)
  alpha = 0.05;
end
rows = size(X,1);cols = size(X,2);

%compute counts for each group
g = unique(group);
for (ii = 1:length(g))
  index(ii) = length(find(group == g(ii)));
end
N = index*cols;
index = [0, index];
index = cumsum(index);

%calculate test statistic for each rows, we will just need to permute
%these values. 
Xs = statwithinsub(X);

%compute our first test stat
for (ii = 1:length(g))
  f = find(group == g(ii));
  T(ii)  = sum(Xs(f));
end
T = statbetweengroups(T, N);
count = 1;

%simulate 
for (ii = 2:iterations)
  t = [];  
  p = randperm(rows);
  for (ii = 1:length(g))
    t(ii) = sum(Xs(p(index(ii)+1:index(ii+1))));
  end
  t = statbetweengroups(t, N);   
  
  if (t >= T)
    count = count + 1;
  end
end

%compute p and return if significant
p  = count / iterations;
if (p < alpha)
  h = 1;
else 
  h = 0;
end


    