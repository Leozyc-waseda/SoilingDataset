% UnpackPyramid - unpack a pyramid image into a cell array
% USAGE:
% unp = UnpackPyramid(pack)
% INPUT:
% pack - A matrix created by the PackPyramid function
% OUTPUT:
% unp - A cell array containing a multiresolution image pyramid, as
%       accepted for input by PackPyramid.
%
function unp = UnpackPyramid(pack)
    packDims = size(pack);
    % Find possible bounds of level 1 image by scaning for sentinel line.
    l1Row = find(isnan(pack(:,1)),1);
    %display(l1Row)
    l1Col = find(isnan(pack(1,:)),1);
    %display(l1Col)

    if(l1Row < packDims(1))
        ss = 1;
        ls = 2;

        % Dimensions of the full resolution image
        baseDim(ss) = l1Row-1;
        baseDim(ls) = packDims(ls);
    else
        ss = 2;
        ls = 1;
        baseDim(ss) = l1Col-1;
        baseDim(ls) = packDims(ls);
    end

    % Copy out the full res image
    unp{1} = pack(1:baseDim(1),1:baseDim(2));

    % Now we basically repeat the same process that was used to pack the
    % original image, except that we copy in the opposite direction.

    lsOff = 1;
    ssOff = baseDim(ss)+2; % Offset one extra to skip sentinel.
    lev = 2;
    lDims = baseDim;
    doneOK = true;
    lo = [0 0];
    hi = [0 0];

    while lsOff < packDims(ls)
        lDims = (lDims + mod(lDims,2))./2;
        lo(ss) = ssOff;
        lo(ls) = lsOff;
        hi(ss) =  lo(ss) + lDims(ss) - 1;
        hi(ls) = lo(ls) + lDims(ls) - 1;

        % We don't know how many levels there were originally, so we try
        % until the dimensions are invalid.
        try
            unp{lev} = pack(lo(1):hi(1),lo(2):hi(2));
            % We assume that if the initial short/long side dimensions are
            % wrong, some NaN values will eventually show up in the
            % pyramid.
            if ~isempty(find(isnan(unp{lev}),1))
                %display(['Found NaN in level ' num2str(lev) ])
                %doneOK = false;
                break
            end
            lsOff = lsOff + lDims(ls);
            lev = lev + 1;
        catch
            break
        end
    end

    % Exit if no problems were detected.
    if doneOK
        %display('UnpackPyramid: Done on first try')
        return
    end
    %display('Retry')
    %pause

    % Evidently we guessed wrong, so swap the short and long sides and try
    % again.
    ss = 2;
    ls = 1;
    baseDim(ss) = l1Col-1;
    baseDim(ls) = packDims(ls);
    %display(baseDim)

    clear unp;

    unp{1} = pack(1:baseDim(1),1:baseDim(2));

    lsOff = 1;
    ssOff = baseDim(ss)+2;
    lev = 2;
    lDims = baseDim;
    doneOK = true;
    while lsOff < packDims(ls)
        %display(lev)
        lDims = (lDims + mod(lDims,2))./2;
        %display(lDims)
        lo(ss) = ssOff;
        lo(ls) = lsOff;
        hi(ss) =  lo(ss) + lDims(ss) - 1;
        hi(ls) = lo(ls) + lDims(ls) - 1;
        %display(hi)
        %display(lo)
        try
            unp{lev} = pack(lo(1):hi(1),lo(2):hi(2));
            %display(unp)
            if ~isempty(find(isnan(unp{lev}),1))
                doneOK = false;
                break
            end
            lsOff = lsOff + lDims(ls);
            lev = lev + 1;
        catch
            break
        end
    end

    if doneOK; return; end
    % If something is still wrong (should not happen) return an empty
    % array.
    unp = {};
