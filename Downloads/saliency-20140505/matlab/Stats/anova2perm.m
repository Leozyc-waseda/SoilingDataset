function stat = anova2perm(X, group, alpha, iterations, ...
                           statwithin, statbetween)
%Computes a two factor permutation test to determine significance for
%main effects of factor1 and factor2. No valid procedure exists to
%compute an interaction as of the writing of this reference. The
%result of each test and the p value is returned in the rows of
%'stats', with factor1 first followed by factor2. X must be a matrix
%where the columns represent levels of the first factor and the rows
%represent measurements*levels of the second factor. Group should be a
%vector of integer values indicating to which level of the second
%factor each row of X belongs.  The default test statitstic is to
%compute the sum of squares of within a factor level sums. This
%statistic (will yield same p-value) as a 'F statistic' from a
%'n-way ANOVA' for equal means). Alpha is the desired significance
%level(default 0.05). 'iterations' is the number of monte-carlo
%simulations to use (default 30,000).The user can optionally supply
%different test statistics for within factor level and between
%levels. This implementation has the restriction that for each of
%factor 2's levels there must be equal subjects in factor 1's
%levels. 
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
  f = find(group == g(ii));
  total(ii) = length(f);
  data{ii} = X(f,:);
end
Nf1 = repmat(rows,1,cols);
Nf2 = total*cols;%number of total measurements for each of factor 2
total = [0, total];
total = cumsum(total);

%compute the first test stat for factor 1
T1 = statwithin(X);
T1 = statbetween(T1, Nf1);
count1 = 1;

%compute our first test stat for factor 2
for (ii = 1:length(data))
  T2(ii) = statwithin(data{ii}(:));
end
T2 = statbetween(T2,Nf2);
count2 = 1;

xt = zeros(size(X));
%simulate 
for (ii = 2:iterations)
  t1 = [];  
  t2 = [];    
  %first factor 1, keep factor 2 constant so shuffle across
  %columns, but keeping the factor2 restricted within level. 
  for (ii = 1:length(data))
    r = randperm(Nf2(ii));
    xt(total(ii)+1:total(ii+1),:) = reshape(data{ii}(r), [], cols);
  end
  t1 = statwithin(xt);
  t1 = statbetween(t1, Nf1);
  
  
  %for factor 2, hold factor 1 constant so shuffle each column
  %seperatly across rows. 
  for (ii = 1:cols)
    r = randperm(rows);
    xt(:,ii) = X(r,ii);
  end
  for ( ii = 1:(length(total)-1) )
    temp = xt(total(ii)+1:total(ii+1), :);
    t2(ii) = statwithin(temp(:));
  end
  t2 = statbetween(t2, Nf2);
  
  if (t1 >= T1)
    count1 = count1 + 1;
  end
  
  if (t2 >= T2)
    count2 = count2 + 1;
  end
end

%compute p and return if significant
stat(1, 2) = count1 / iterations;
if (stat(1,2) < alpha)
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






    