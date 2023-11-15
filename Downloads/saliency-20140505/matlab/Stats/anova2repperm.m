function stat = anova2repperm(X, group, alpha,iterations, ...
                              statwithin, statbetween)
%Computes a two factor permutation test where factor 1 is a
%repeated. Significance for main effects of factor1, factor2 and the
%interaction are returned in the rows of 'stats'. X must be a matrix
%where the columns represent levels of the first factor (repeated
%measure) and the rows represent measurements*levels of the second
%factor. Group should be a vector of integer values indicating to
%which level of the second factor each row of X belongs. 'Repeat'
%should be 0 for no repeated measures or 1 if the columns are a
%repeated mesures factor. The default test statitstic is to compute
%the sum of squares of within a factor level sums. This statistic
%(will yield same p-value) as a 'F statistic' from a 'n-way ANOVA'
%for equal means). Alpha is the desired significance level(default
%0.05). 'iterations' is the number of monte-carlo simulations to use
%(default 30,000).The user can optionally supply different test
%statistics for within subject or factor level and between levels. 
%
%references:
%From Edington, ES. (1995). Randomization tests. New York, NY:Marcel
%Dekker inc.
%
%written by David J. Berg (dberg@usc.edu) 

if (nargin < 6)
  statbetween=@(x, N)(sum( (x.^2)./N )) ;
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

rows = size(X,1);cols = size(X,2);

%compute counts for each group
g = unique(group);
for (ii = 1:length(g))
  index(ii) = length(find(group == g(ii)));
end
Nf2 = index*cols;
Nf1 = ones(1,cols) * rows;
index = [0, index];
index = cumsum(index);

%compute test statistic for repeated measure
T1 = statwithin(X);
T1 = statbetween(T1, Nf1);
count1 = 1;

%compute test statistic for main effect of factor2
Xs = statwithin(X');
for (ii = 1:length(g))
  f = find(group == g(ii));
  T2(ii) = sum(Xs(f));
end
T2 = statbetween(T2, Nf2);
count2 = 1;

%compute all the parwise column difference for our interaction test
D = [];
for (ii = 1:cols-1)
  for (jj = ii+1:cols)
    D = [D,X(:,ii) - X(:,jj)];
  end
end
ND = rows .* size(D,2);
if (size(D,2) > 1)
  D = statwithin(D');
end
for (ii = 1:length(g))
  f = find(group == g(ii));
  TD(ii) = sum(D(f));
end
TD = statbetween(TD, ND);
countd = 1;

%simulate
xt = zeros(size(X));
for (ii = 2:iterations)%start at 2 because our original counts as 1 permutation

  %compute for factor 1
  for (jj = 1:rows)
    xt(jj,:) = X(jj,randperm(cols));
  end
  t1 = statwithin(xt);
  t1 = statbetween(t1, Nf1);
  
  %compute for factor 2
  p = randperm(rows);
  for (ii = 1:length(g))
    t2(ii) = sum(Xs(p(index(ii)+1:index(ii+1))));
  end
  t2 = statbetween(t2, Nf2);
  
  %compute for factor 2
  p = randperm(rows);
  for (ii = 1:length(g))
    td(ii) = sum(D(p(index(ii)+1:index(ii+1))));
  end
  td = statbetween(td, ND);
  
  %test and increment counter
  if (t1 >= T1)
    count1 = count1 + 1;
  end
  
  %test and increment counter
  if (t2 >= T2)
    count2 = count2 + 1;
  end
  
  %test and increment counter
  if (td >= TD)
    countd = countd + 1;
  end
end

%compute p and return if significant
stat(1, 2) = count1 / iterations;
if (stat(1, 2) < alpha)
  stat(1, 1) = 1;
else 
  stat(1, 1) = 0;
end

stat(2, 2) = count2 / iterations;
if (stat(2,2) < alpha)
  stat(2, 1) = 1;
else 
  stat(2, 1) = 0;
end

stat(3, 2) = countd / iterations;
if (stat(3, 2) < alpha)
  stat(3, 1) = 1;
else 
  stat(3, 1) = 0;
end




    