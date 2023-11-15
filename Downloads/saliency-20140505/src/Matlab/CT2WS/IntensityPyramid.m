% IntensityPyramid - Create a multi-resolution image pyramid.
function ip = IntensityPyramid(img, levels)
    if ~exist('levels','var'); levels=[]; end
    %filter = ones(4,4)./16;
    filter = Gaussian(16,16,16/8);
    if isempty(levels); levels = 7; end
    pyr = Pyramid(img, filter, levels);
    ip = PackPyramid(pyr);

%     [pr pc] = size(ip);
%     [ir ic] = size(img);
%     ip(pr, pc-1) = ir;
%     ip(pr, pc) = ic;
    %nanIx = find(isnan(ip));
    ip(isnan(ip)) = -10000;
