function f = binomialFilter(halfsiz)
%function f = binomialFilter(halfsiz)
% returns a binomial filter of unit sum and size 2*halfsiz + 1

for ii = 1:2*halfsiz+1
  f(ii) = nchoosek(2*halfsiz, ii-1);
end
f = f / sum(f);
