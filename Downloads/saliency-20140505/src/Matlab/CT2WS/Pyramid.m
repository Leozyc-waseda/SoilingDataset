function pyr = Pyramid(base, filter, levels)
    % First level is input base image
    pyr{1} = base;
    if levels < 1
        return
    end
    for l = 2:(levels+1)
        im0 = pyr{l-1};
        [r0 c0] = size(im0);
        % Pad image to even dimensions
        if mod(r0,2)
            r0 = r0+1;
            im0(r0,:) = 0;
        end
        if mod(c0,2)
            c0 = c0+1;
            im0(:,c0) = 0;
        end
        % Filter the image
        im1 = imfilter(im0, filter);
        % Decimate 2:1
        [r c] = size(im1);
        pyr{l} = im1(1:2:r, 1:2:c);
    end