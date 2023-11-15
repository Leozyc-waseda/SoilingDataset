function [ip pyr]= OrientationPyramid(img, levels, angle)
    if ~exist('levels','var'); levels=[]; end
    fr = 16;
    period = fr;
    sigma = 3*fr/2;
    filter = Gabor(fr, fr, angle, period, sigma);
    %figure;imagesc(filter);colormap(gray);
    if isempty(levels); levels = 7; end
    pyr = Pyramid(img, filter, levels);
    ip = PackPyramid(pyr);
    ip(isnan(ip)) = -10000;
