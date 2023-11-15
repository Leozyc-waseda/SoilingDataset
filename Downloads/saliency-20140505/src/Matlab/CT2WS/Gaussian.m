% Gaussian - creates a Gaussian filter kernal
%
% USAGE:
% g  = Gaussian(r, c, sigma)
%
% INPUTS:
% r, c - row and column dimensions of kernel
% sigma - standard deviation of Gaussian
%
% OUPUT
% g - A symmetric Gaussian kernel, centered in the output matrix.
%
function g = Gaussian(r, c, sigma)
    xc = (1+c)/2;
    yc = (1+r)/2;
    x = repmat(1:c, r, 1) - xc;
    y = repmat(1:r, c, 1)' - yc;
    g = exp(-(x.*x + y.*y)./(2*sigma*sigma));
    g = g./sum(g(:));