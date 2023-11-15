function Calib = findFullCalib(sc, ey)
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
% eyc is the calibrated eye regard that comes out

OUTDIST = 50;                 % outlier distance in pixels
maxresi = 35;  % max acceptable residual distance, ppd is 35 pixel/degree for x, 33 pixel/degree for y

% create custom struct for calibration
Calib = struct('Mpre',eye(3), 'Mpost', eye(3), ...
	       'tpsInfo', [], 'tpsc', 0.25);

% Let's start with a pre-TPS affine calibration using all points:
MpreOld = Calib.Mpre; % keep old calibration in case we abort
Calib.Mpre = affineCalibration(sc, ey);

eyc1 = calibAffine(ey, Calib.Mpre);

% let's eliminate the outliers:
[idx,res] = findOutliers(sc, eyc1, OUTDIST);
%figure(4)
%plot(sc(1,:), sc(2,:),'bx', eyc1(1,:),eyc1(2,:),'r+');
%drawnow;

% show the residual distances:
subplot(2, 2, 1); plotCalib(sc, eyc1); title('after preAff');

% NB: findOutliers returns all idx that are NOT outliers

if (length(idx) < 5 & ~isempty(Calib.tpsInfo)) 
  %if we have a fishy calibration
  %but have a preexisting good thin spline/Mpre
  disp('not enough good points - keeping Mpre and TPS');
  Calib.Mpre = MpreOld;
  
  eyc1 = calibAffine(ey, Calib.Mpre);         % pre-TPS affine transform
  eyc2 = deformPoints(eyc1', Calib.tpsInfo, [], Calib.tpsc)'; % thin-plate-spline warping
  sc = calibAffine(eyc2, Calib.Mpost);       % post-TPS affine transform


else
  if (length(idx) < 6)% this calib is fishy!
    % output points from best to worst
    
    disp('Residuals report:');
    [foo,ibest] = sort(res);
    for ii = 1:length(sc)
      fprintf(1,'#%d, res(%g,%g)->%g. \n', ...
	      ibest(ii), sc(1,ibest(ii)), ...
	      sc(2,ibest(ii)),res(ibest(ii)));
    end
    error('findFullCalib:badCal', ['Fishy calibration -' ...
		    ' too few good points (n = %d)!'],length(idx));
  end
  
  % extract only good calibration points
  if(length(idx) < length(ey)) 
    disp('Eliminating outliers in pre-Affine');
    sc = sc(:, idx); ey = ey(:, idx);
  end
  
  % try just an affine calibration
  Calib.Mpre = affineCalibration(sc, ey);
  eyc1 = calibAffine(ey, Calib.Mpre);
  
  % full TPS calibration:
  % compute the TPS calibration between eyc and sc:
  landmarks = [eyc1' sc'];  % target and template
  Calib.tpsInfo = calcTPSInfo(landmarks);
  
  % get the deformed points for plotting:
  eyc2 = deformPoints(eyc1', Calib.tpsInfo, [], Calib.tpsc)'; % warp
  
  % for now, Mpost is identity; it will come later if we have
  % some post-calibration data:
  Calib.Mpost = eye(3);
  eyc = eyc2;
end

% show the residual distances:
subplot(2, 2, 1); plotCalib(sc, eyc1); title('after preAff');
subplot(2, 2, 2); plotCalib(sc, eyc2); title('after TPS');
subplot(2, 2, 3); plotCalib(sc, eyc); title('after postAff');
subplot(2, 2, 4); plotMesh(ey, Calib); title('pre+TPS+post');

% finding the maximum and median errors for horiz and vert components
resi1 = max(max(abs(sc - eyc)));
resi2 = mean(mean(abs(sc - eyc)));

% check residual values 
% 0.5 value is arbitrary
if (resi1 > maxresi | resi2 > maxresi * 0.5 | isnan(resi1) | isnan(resi2))
  error('findFullCalib:badCal', 'Residual Values (manhattan) are too large');
end

fprintf(1, 'max residuals is %.2f, mean residuals is %.2f\n', resi1, resi2);
