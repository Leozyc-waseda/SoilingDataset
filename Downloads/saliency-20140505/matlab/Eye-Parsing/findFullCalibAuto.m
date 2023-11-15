function Calib = findFullCalibAuto(scIdeal, eYfixData, used, center)
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
% center is the center point, 1x2

OUTDIST = 50;                 % outlier distance in pixels, abt 1.5 degs
maxresi = 35;  % max acceptable residual distance, ppd is 35
% pixel/degree for x, 33 pixel/degree for y
snapdist = 30;
Npts = length(used);

% create custom struct for calibration
Calib = struct('Mpre',eye(3), 'Mpost', eye(3), ...
    'tpsInfo', [], 'tpsc', 0.25);

% Let's start with a pre-TPS affine calibration using all points:
MpreOld = Calib.Mpre; % keep old calibration in case we abort
goodCalib = false; errmsg = '';
mouse_prompt = false;

nPossCalibs = 2^Npts;
NEXACT = 50000;
if nPossCalibs > NEXACT
    [sc ey] = getCalibCoords(scIdeal,eYfixData,used);
    Calib = findFullCalib(sc,ey);
    disp('Too many calibration points for exact test: exiting')
    return;
end
isGoodCalib = false(1,nPossCalibs);
worstres = zeros(1,nPossCalibs);
avgres = zeros(1,nPossCalibs);

for i = 0:nPossCalibs-1
    ii=i+1;
    used = (dec2bin(i)=='0'); % start from all ones and move down
    if sum(used)<5 
      continue; % we want at least 5 preselected calibration points 
    end
    [isGoodCalib(ii),Calib] = tryCalib(scIdeal,eYfixData,used);

    % finding the maximum and median errors for horiz and vert components
    if isGoodCalib(ii) % if we have full-calibrated data
      [scGood eyGood] = getCalibCoords(scIdeal,eYfixData,used);

      [eycPre eycSpline eycPost] = execCalib(eyGood);
      resi1 = max(max(abs(scGood - eycPost)));
      resi2 = mean(mean(abs(scGood - eycPost)));
      worstres(ii)=resi1;
      avgres(ii)=resi2;
      %resmanhat = resx + resy;
      
      % check residual values
      % 0.5 value is arbitrary
      if (resi1 > maxresi || resi2 > maxresi * 0.5 || isnan(resi1) || isnan(resi2))
	isGoodCalib(ii) = false;
      end
    end
end

% pick the best calibrations by various criteria
[foo, ib_worst] = min(worstres);
[foo, ib_avg] = min(avgres);
[foo, ib_combo] = min(worstres+avgres);

% report the summary of residual distances:
fprintf(['(1) best worst case: (npts = %n),\n '...
	'max residuals is %.2f, mean residuals is %.2f\n'],...
	sum(dec2bin(ib_worst)=='0',worstres(ib_worst),avgres(ib_worst));
fprintf(['(2) best avg case: (npts = %n),\n '...
	'max residuals is %.2f, mean residuals is %.2f\n'],...
	sum(dec2bin(ib_avg)=='0',worstres(ib_avg),avgres(ib_avg));
fprintf(['(3) best combined case: (npts = %n),\n '...
	'max residuals is %.2f, mean residuals is %.2f\n'],...
	sum(dec2bin(ib_combo)=='0',worstres(ib_combo),avgres(ib_combo));
r = 0;
i_sel = [ib_worst, ib_avg, ib_combo];
while r < 1 || r > 3
  i_pick=-1;
  r = input('Which calibration would you select? ', 's')
  r = str2num(r(1));
  if  r > 1 && r < 3'
    i_pick=i_sel(str2num('1');
  end
  if i_pick~=-1
    [foo,Calib]=
  end

end

function [isgood, Calib] = tryCalib(scI, eyF, used);
% create custom struct for calibration
Calib = struct('Mpre',eye(3), 'Mpost', eye(3), ...
    'tpsInfo', [], 'tpsc', 0.25);

[scGood eyGood] = getCalibCoords(scI,eyF,used);
    
Calib.Mpre = affineCalibration(scGood, eyGood);

% given selected points, find the calibration
eycPre = calibAffine(eyGood, Calib.Mpre);
eycSpline = [];
eycPost = [];

% let's eliminate the outliers:
[idx,res] = findOutliers(scGood, eycPre, OUTDIST);
inlier = ismember(1:Npts,idx);   
% NB: findOutliers returns all idx that are NOT outliers

if (length(inlier) < 6)% this calib is fishy - less than 6 good pts!
  isgood = false;
  Calib.tpsInfo = [];
else % seems OK - try the calibration
     % extract only good calibration points
     if(sum(inlier) < sum(used))
       disp('Eliminating outliers in pre-Affine');
       scGood = scGood(:, inlier); eyGood = eyGood(:, inlier);
     end
     
     % try just an affine calibration
     Calib.Mpre = affineCalibration(scGood, eyGood);
     eycPre = calibAffine(eyGood, Calib.Mpre);
     
     % full TPS calibration:
     % compute the TPS calibration between eyc and sc:
     landmarks = [eycPre' scGood'];  % target and template
     Calib.tpsInfo = calcTPSInfo(landmarks);
     
     % get the deformed points for plotting:
     eycSpline = deformPoints(eycPre', Calib.tpsInfo, [], Calib.tpsc)'; % warp
     
     % for now, Mpost is identity; it will come later if we have
     % some post-calibration data:
     Calib.Mpost = eye(3);
     eycPost = eycSpline; % equiv to eycPost = calibAffine(eycSpline, Calib.Mpost);
     isgood = true;
end

    function [ey1, ey2, eyf] = execCalib(ey) % ey must be 2xN
        % uses value Calib
        ey1 = calibAffine(ey,Calib.Mpre);
        ey2 = []; eyf = [];
        if TPSisSet();
            ey2 = deformPoints(ey1', Calib.tpsInfo, [], Calib.tpsc)';
            eyf = calibAffine(ey2,Calib.Mpost);
        end
    end

    function d = dist(P1,P2)
        % p1 and p2 are 2x1
        d = sqrt((P1(1)-P2(1))^2+(P1(2)-P2(2))^2);
    end

    function ret = TPSisSet()
        ret = (~isempty(Calib.tpsInfo));
    end

    function [scUsed, eyUsed] = getCalibCoords(scAll, eyAll, ...
						      isUsed)
    % center is always the first coordinate
        scUsed = [center scAll(:,isUsed)];
        eyUsed = [mean(eyAll(1:2,isUsed),2) eyAll(3:4,isUsed)];
        % assumes center is not in inUsed      
    end

    function [scReg, eyReg] = getAllCoords(scAll, eyAll)
        % get all coords
        isUsed = true(1,Npts);
        for ii = 1:Npts
            if all(scAll(:,ii) == center)
                isUsed(ii) = false;
            end
        end
        [scReg, eyReg] = getCalibCoords(scAll,eyAll,isUsed);
    end

    function plotAllCalib(sc, ey, u)
        u = [false; u]'; % for the center point
        badFixes = all(sc==0,1);
        
        % build some vectors going from screen to eye:
        p = NaN(2, 3 * sum(u));
        p(:, 1:3:3*sum(u)) = sc(:,u);
        p(:, 2:3:3*sum(u)) = ey(:,u);
        
        plot(p(1, :), p(2, :), 'b');
        hold on;
        plot(sc(1, u), sc(2, u), 'b*');
        
        plot(sc(1,~u & ~badFixes),sc(2,~u & ~badFixes), 'ro');
        hold off;
        
    end
end