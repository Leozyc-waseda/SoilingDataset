function [r c] = PyramidLevelDims(br, bc, levels)
    %display(['PyramidLevelDims: ' num2str(br) ',' num2str(bc) ',' num2str(levels)])
    % First level is input base image
    r = br;
    c = bc;
    if levels < 1
        return
    end
    for l = 2:(levels+1)
        % Pad image to even dimensions
        r = r + mod(r, 2);
        c = c + mod(c, 2);
        r = r/2;
        c = c/2;
        %display(['PyramidLevelDims: ' num2str(r) ',' num2str(c) ])
    end

