function [r c] = PyramidLevelSizes(br, bc, levels)
    %display(['PyramidLevelDims: ' num2str(br) ',' num2str(bc) ',' num2str(levels)])
    % First level is input base image
    r(1) = br;
    c(1) = bc;
    if levels < 1
        return
    end
    for l = 2:(levels+1)
        % Pad image to even dimensions
        r(l) = r(l-1) + mod(r(l-1), 2);
        c(l) = c(l-1) + mod(c(l-1), 2);
        r(l) = r(l)/2;
        c(l) = c(l)/2;
        %display(['PyramidLevelDims: ' num2str(r) ',' num2str(c) ])
    end

