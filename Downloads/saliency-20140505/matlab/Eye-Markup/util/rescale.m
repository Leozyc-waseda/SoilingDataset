function y = rescale(x,a,b)

% rescale - rescale data in [a,b]
%
%   y = rescale(x,a,b);
%
%   Copyright (c) 2004 Gabriel Peyré

if nargin<2
    a = 0;
end
if nargin<3
    b = 1;
end

m = min(x,[],2);
M = max(x,[],2);

y = repmat(b-a,size(x,1),length(x)) .* (x-repmat(m,1,length(x)))./...
    repmat(M-m,1,length(x)) + repmat(a,size(x,1),length(x));

