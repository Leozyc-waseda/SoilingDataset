function M = shiftAffine(M, vec)
% shifts calibration so that it is centered on a new coord
    if all(size(M)==[3 3]) && numel(vec) == 2 % must have right vals
        M(1,3) = M(1,3)-vec(1);
        M(2,3) = M(2,3)-vec(2);
    end
    
        