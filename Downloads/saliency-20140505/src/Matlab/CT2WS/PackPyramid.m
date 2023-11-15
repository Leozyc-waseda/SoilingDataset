% PackPyramid - Pack a cell array image pyramid into a matrix
% USAGE:
% pack =  PackPyramid(p)
% INPUT:
% p - a cell array containing an image pyramid. p{1} is the
%     original finest resolution image, and successive cells
%     hold increasingly coarse resolution images.
% OUTPUT:
% pack - a matrix containing all of the pyramid images, tiled
%        compactly into a single image.
%
function pack = PackPyramid(p)
    nLevels = length(p);
    baseDim = size(p{1});
    % Idenitify the long and short sides of the base image.
    [ms ss] = min(baseDim);
    [ml ls] = max(baseDim);
    if ms == ml
        ss = 1;
        ls = 2;
    end

    % Calculate dimensions of the packed image. Note that '1' is added to
    % the 'short' side dimension to allow for insertion of a line of
    % sentinal values which will be needed to unpack.
   packL = 0;
    for lev = 2:nLevels
        lDims = size(p{lev});
        packL = packL + lDims(ls);
    end
    packS = baseDim(ss)+size(p{2},ss)+1;
    packL = max(packL, baseDim(ls));
    packDims(ss) = packS;
    packDims(ls) = packL;
    pack = NaN(packDims);

    lsOff = 1;
    % Initially increment ssOff by one extra to leave a sentinel line.
    ssOff = baseDim(ss)+2;
    pack(1:baseDim(1),1:baseDim(2)) = p{1};

    for lev = 2:nLevels
        lDims = size(p{lev});
        lo(ss) = ssOff;
        lo(ls) = lsOff;
        hi(ss) =  lo(ss) + lDims(ss) - 1;
        hi(ls) = lo(ls) + lDims(ls) - 1;
        %figure;imagesc(p{lev});colormap(gray);axis equal;title(['PackPyramid: level=' num2str(lev)])
        pack(lo(1):hi(1),lo(2):hi(2)) = p{lev};
        lsOff = lsOff + lDims(ls);
    end

