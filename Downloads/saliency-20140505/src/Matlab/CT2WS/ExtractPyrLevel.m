function img = ExtractPyrLevel(pyrImg, level)
%     class(pyrImg)
%     display(level)
%     display(min(min(pyrImg)))
    pyrImg(pyrImg==-10000) = NaN;
%     foo = zeros(size(pyrImg));
%     foo(isnan(pyrImg)) = 100;
%     figure;imagesc(foo)
%     figure;imagesc(pyrImg);colormap(gray)
    pyr = UnpackPyramid(pyrImg);
    if isempty(pyr)
        error('ExtractPyrLevel: could not unpack')
    end
    %class(pyr)
    img = pyr{level+1};
    img(isnan(img)) = 0;
%     figure;imagesc(img);colormap(gray)