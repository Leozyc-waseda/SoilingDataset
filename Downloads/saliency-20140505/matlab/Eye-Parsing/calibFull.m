function sc = calibFull(ey, calib)
% This function runs the calibration with pre-affine, thin-spline,
% and post-affine calibrations.
% The calib structure is of the following:
% CALIB.Mpre,Mpost is a 3x3 representing affine transformation on [x y 1];
% CALIB.tpsInfo is a structure:
%    tpsInfo has the form
%    tpsInfo.points   The XY location of all of the target points (Mx2)
%    tpsInfo.a        A 3x1 vector of the affine transformation info.
%    tpsInfo.w        A Mx1 vector containing the weights for each of the points
% CALIB.tpsc is a fraction showing the fraction contributing to the
% tps...

eyc1 = calibAffine(ey, calib.Mpre);         % pre-TPS affine transform
eyc2 = deformPoints(eyc1', calib.tpsInfo, [], calib.tpsc)'; % thin-plate-spline warping
sc = calibAffine(eyc2, calib.Mpost);       % post-TPS affine transform

