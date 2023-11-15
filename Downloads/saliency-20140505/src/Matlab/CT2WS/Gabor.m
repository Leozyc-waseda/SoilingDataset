% Gabor - creates a Gabor function filter kernel
%
% USAGE:
% g = Gabor(r, c, theta, period, sigma)
%
% INPUTS:
% r, c - row and column dimensions of the output
% theta - angle (in degrees) of the cosine component.
% period - period (in samples) of the cosine
% sigma - standard deviation of the symetric Gaussian component
%
% OUTPUT:
% g - a Gaussian weighted cosine wave centered in the output matrix.
%     Normalized so that the elements sum to 1.
%
function g = Gabor(r, c, theta, period, sigma)
    theta = pi*theta/180;
    cs = cos(theta);
    sn = sin(theta);
    xc = (1+c)/2;
    yc = (1+r)/2;
    x = repmat(1:c, r, 1) - xc;
    y = repmat(1:r, c, 1)' - yc;
    g = cos((2*pi/period)*(cs*x + sn*y));
    gw = exp(-(x.*x + y.*y)./(2*sigma*sigma));
    g = g.*gw;
    g = g./sum(sum(g));