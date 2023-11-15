% UpsamplePyrLevel - Upsample a coarser pyramid image to a higher res
% USAGE:
% sImg = UpsamplePyrLevel(sImg, cLev, sLev, pR, pC)
% INPUTS:
% sImg - the input "surround" image.
% cLev - pyramid level of the "center" image'. Indicates the degree to
%        which the input image is to be upsampled.
% sLev - pyramid level of the input image. Higher level = coarser
%        resolution, so require sLev >= cLev. (Level = 0 is original image)
% pR, pC - rows and cols of the original image.
%

function sImg = UpsamplePyrLevel(sImg, cLev, sLev, pR, pC)
    if sLev < cLev; error('UpsamplePyrLevel: inconsistent pyramid levels'); end
    [lvlR lvlC] = PyramidLevelSizes(pR, pC, sLev);

    while sLev > cLev
        sImg = kron(sImg, ones(2,2));
        sImg = sImg(1:lvlR(sLev),1:lvlC(sLev));
        sLev = sLev - 1;
    end
