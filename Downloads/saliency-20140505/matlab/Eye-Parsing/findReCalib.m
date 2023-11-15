function Calib = findReCalib(sc, ey, Calib)
% This function finds a calibration to map ey to sc 
% with pre-affine, thin-spline, and post-affine calibrations.
% The structure it returns is of the following:
% CALIB.Mpre,Mpost is a 3x3 representing affine transformation on [x y 1];
% CALIB.tpsInfo is a structure:
%    tpsInfo has the form
%    tpsInfo.points   The XY location of all of the target points (Mx2)
%    tpsInfo.a        A 3x1 vector of the affine transformation info.
%    tpsInfo.w        A Mx1 vector containing the weights for each of the points
% CALIB.tpsc is a fraction showing the fraction contributing to the
% tps...

OUTDIST = 100;                 % outlier distance in pixels
maxresi = 35;  % max acceptable residual distance, ppd is 35 pixel/degree for x, 33 pixel/degree for y

% quick affine post-calibration: let's first get the points
% transformed by the preAff and the TPS:
MpostOld = Calib.Mpost;
eyc1 = calibAffine(ey, Calib.Mpre);
eyc2 = deformPoints(eyc1', Calib.tpsInfo, [], Calib.tpsc)';

% try attaching the new calibration
Calib.Mpost = affineCalibration(sc, eyc2);
eyc = calibAffine(eyc2, Calib.Mpost);

% let's eliminate the outliers:
idx = findOutliers(sc, eyc, OUTDIST);
% NB: findOutliers returns all idx that are NOT outliers

if (length(idx) < 4)
  disp('not enough good points - keeping Mpost');
  Calib.Mpost = MpostOld;
  
  %redo w/ the old one
  eyc = calibAffine(eyc2, Calib.Mpost);
else  
  disp('Eliminating outliers in post-Affine');
  sc = sc(:, idx); ey = ey(:, idx);

  % try without the outliers
  eyc1 = calibAffine(ey, Calib.Mpre);
  eyc2 = deformPoints(eyc1', Calib.tpsInfo, [], Calib.tpsc)';
  Calib.Mpost = affineCalibration(sc, eyc2);
  eyc = calibAffine(eyc2, Calib.Mpost);
end

% show the residual distances:
subplot(2, 2, 1); plotCalib(sc, eyc1); title('after preAff');
subplot(2, 2, 2); plotCalib(sc, eyc2); title('after TPS');
subplot(2, 2, 3); plotCalib(sc, eyc); title('after postAff');
subplot(2, 2, 4); plotMesh(ey, Calib); title('pre+TPS+post');

% finding the maximum and median errors for horiz and vert components
resmax = max(max(abs(sc - eyc)));
resmed = mean(mean(abs(sc - eyc)));

% check residual values 
% 0.5 value is arbitrary
if (resmax > maxresi | resmed > maxresi * 0.5 | isnan(resmax) | isnan(resmed))
  % output points from best to worst

  resx = abs(sc(1,:)-eyc(1,:));
  resy = abs(sc(2,:)-eyc(2,:));
  resmanhat = resx + resy;
  
  fprintf('Residuals report: maxAllowed = %g\n',maxresi);
  [foo,ibest] = sort(resmanhat);
  for ii = 1:length(sc)
      fprintf(1,'#%d, res(%g,%g)->(%g,%g0) \n', ...
	      ibest(ii), sc(1,ibest(ii)), ...
	      sc(2,ibest(ii)),resx(ibest(ii)),resy(ibest(ii)));
  end
  error('findReCalib:badCal', 'Residual Values (manhattan) are too large');
end

fprintf(1, 'max residuals is %.2f, mean residuals is %.2f\n', resmax, resmed);
