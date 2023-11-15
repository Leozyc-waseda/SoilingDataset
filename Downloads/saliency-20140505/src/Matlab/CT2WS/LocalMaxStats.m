% LocalMaxStats - Local Maximum Statistics for an image.
%
% USAGE:
% avg = LocalMaxStats(img)
%
% INPUTS:
% img - an image matrix
%
% OUTPUTS:
% avg - the average of the local maximum values in the image (with the
%       global maximum excluded). A pixel is considered a local maximum if
%       its values exceeds all eight of its immediate neighbors.
%
function avg = LocalMaxStats(img)
    [r c] = size(img);
    nLocMax = 0;
    avg = 0;
    extreme = -inf;
    for rIx = 2:(r-1)
        for cIx = 2:(c-1)
            ctr = img(rIx, cIx);
            nbr = img(rIx+(-1:1), cIx+(-1:1));
            cmp = ctr > nbr;
            if sum(cmp(:)) > 7
                nLocMax = nLocMax + 1;
                avg = avg + ctr;
            end
            extreme = max(extreme,ctr);
        end
    end
    avg = avg - extreme;
    if nLocMax > 0
        avg = avg / nLocMax;
    end



