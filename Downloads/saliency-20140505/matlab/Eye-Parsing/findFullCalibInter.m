function Calib = findFullCalibInter(scIdeal, eYfixData, user_selected, CENTER)
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
Npts = length(user_selected);
% create custom struct for calibration

Calib = struct('Mpre',eye(3), 'Mpost', eye(3), ...
    'tpsInfo', [], 'tpsc', 0.25);
% Let's start with a pre-TPS affine calibration using all points:
MpreOld = Calib.Mpre; % keep old calibration in case we abort
goodCalib = false; errmsg = '';
mouse_prompt = false;

user_toggled = user_selected';
while ~goodCalib
    %reminder: user_selected refers only to the original ordering data
    [scGood eyGood] = getCalibCoords(scIdeal,eYfixData,user_toggled);
    
    Calib.Mpre = affineCalibration(scGood, eyGood);
    
    % UNDO: do calibrations on full data
    eycPre = calibAffine(eyGood, Calib.Mpre);
    
    % let's eliminate the outliers:
    [idx,res] = findOutliers(scGood, eycPre, OUTDIST);
    
    in_use = ismember(1:Npts,idx);
    % show the residual distances:
    %subplot(2, 2, 1); plotCalib(scGood, eycPre); title('after preAff');
    
    % NB: findOutliers returns all idx that are NOT outliers
    
    if (sum(in_use) < 5 && TPSisSet())
        %if we have a fishy calibration
        %but have a preexisting good thin spline/Mpre
        disp('not enough good points - keeping Mpre and TPS');
        Calib.Mpre = MpreOld;
        
        % do the entire calib in one shot
        [foo bar eycPost] = execCalib(eyGood);
        
    elseif (sum(in_use) < 6)% this calib is fishy - less than 6 good pts!
        errmsg = sprintf(' too few good points (n = %d)',length(idx));
        goodCalib = false;
    else % try the calibration
        % extract only good calibration points
        if(sum(in_use) < sum(user_toggled))
            disp('Eliminating outliers in pre-Affine');
        end
        % try just an affine calibration
        Calib.Mpre = affineCalibration(scGood(:,in_use), eyGood(:,in_use));
        
        eycPre = calibAffine(eyGood(:,in_use), Calib.Mpre);
        
        % full TPS calibration:
        % compute the TPS calibration between eyc and sc:
        landmarks = [eycPre' scGood(:,in_use)'];  % target and template
        Calib.tpsInfo = calcTPSInfo(landmarks);
        
        % get the deformed points for plotting:
        eycSpline = deformPoints(eycPre', Calib.tpsInfo, [], Calib.tpsc)'; % warp
        
        % for now, Mpost is identity; it will come later if we have
        % some post-calibration data:
        Calib.Mpost = eye(3);
        eycPost = eycSpline; % equiv to eycPost = calibAffine(eycSpline, Calib.Mpost);
        goodCalib = true;
    end
    
    % finding the maximum and median errors for horiz and vert components
    
    if TPSisSet() % if we have full-calibrated data
        resi1 = max(max(abs(scGood(:,in_use) - eycPost)));
        resi2 = mean(mean(abs(scGood(:,in_use) - eycPost)));
        %resmanhat = resx + resy;
        if (resi1 > maxresi || resi2 > maxresi * 0.5 || isnan(resi1) || isnan(resi2))
            goodCalib = false;
            errmsg = 'Residual Values (manhattan) are too large: retry calibration';
        end
        % report the summary of residual distances:
        %fprintf(1, 'max residuals is %.2f, mean residuals is %.2f\n', resi1, resi2);
        %else
        %    goodCalib = false;
        %    errmsg = 'Not enough good points: retry calibration';
    end
    
    % check residual values
    % 0.5 value is arbitrary
    
    if ~goodCalib        
        if ~mouse_prompt %if we haven't shown this before
            mouse_prompt = true;
            disp('Left-click to toggle point, press enter to quit...')
            disp('Starting mouse input:')
            %disp('right mouse views eyetrace data') currently not enabled
        end
        plotInterCalib(scIdeal,eYfixData,Calib,user_toggled);
        [xclick, yclick, but] = getClick();
        % TODO: case out but = 1 (left), but = 3 (right), but = [] (enter)
        % quit case
        quitOut = isempty(but);
        if quitOut
            break;
        end
        
        % left-click case
        snappt = 0;
        for jj = 1:Npts
            if dist(scIdeal(:,jj),[xclick yclick]) < snapdist && all(scIdeal(:,jj)~=0)
                snappt = jj;
                break;
            end
        end
        if snappt>0
            user_toggled(snappt) = ~user_toggled(snappt);
            if sum(user_toggled) == 0 % we need at least some points
                user_toggled(snappt) = ~user_toggled(snappt);
            end
        end
    end % if ~goodCalib
    
    if goodCalib
        fprintf('Calibration is GOOD... (n kept = %d)\n', sum(in_use));
        plotInterCalib(scIdeal,eYfixData,Calib,user_toggled);
        % report the summary of residual distances:
        fprintf(1, 'max residuals is %.2f, mean residuals is %.2f\n', resi1, resi2);
        in = input('Accept calibration? [RET = yes]','s');
        if (~isempty(in) && lower(in(1)) ~= 'y')
            goodCalib = false;
        end
        
    end
    
end %while badCalib loop
if ~goodCalib
    error('findFullCalibInter:badcal', 'Bad calibration: %s',errmsg);
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

    function plotInterCalib(scOrig,eyOrig,Calib, do_tryCalib)
        % move center pt to front, remove elsewhere if it exists
        is_cent = zeros(1,Npts); is_zero = zeros(1,Npts);
        for ii = 1:Npts
            is_cent(ii) = all(scOrig(:,ii)==CENTER);
            is_zero(ii) = all(scOrig(:,ii)==0);
        end
        scCfirst = [CENTER scOrig(:,~is_cent)];
        eyCfirst = [mean(eyOrig(1:2,do_tryCalib & ~is_cent),2) ...
            eyOrig(3:4,~is_cent)];
        do_tryCalib = [true do_tryCalib(~is_cent)];
        % insert zeros where scCfirst is 0 (or maybe at the center)
             
        % run calibration only on good points
        [ey1 ey2 eyf] = execCalib(eyCfirst(:,do_tryCalib));
        
        % rearrange the ey's out to align with
        sc_Pre = zeros(2,Npts+1); sc_Pre(:,do_tryCalib) = ey1;

        idx = findOutliers(scCfirst, sc_Pre, OUTDIST);
        is_good = ismember(1:size(sc_Pre,2),idx);
        
        % set goodness status of center point
        
        % pass user_toggled points and fit_points
        subplot(2, 2, 1); plotAllCalib(scCfirst, sc_Pre,do_tryCalib,is_good); title('after preAff');
        if TPSisSet()
            sc_Spl = zeros(2,Npts+1); sc_Spl(:,do_tryCalib) = ey2;
            sc_Post = zeros(2,Npts+1); sc_Post(:,do_tryCalib) = eyf;
            is_good(1) = ( dist(sc_Post(:,1),CENTER) < maxresi ); % update goodness of center
            
            subplot(2, 2, 2); plotAllCalib(scCfirst, sc_Spl,do_tryCalib,is_good); title('after TPS');
            subplot(2, 2, 3); plotAllCalib(scCfirst, sc_Post,do_tryCalib,is_good); title('after postAff');
            subplot(2, 2, 4); plotMesh(eyf, Calib); title('pre+TPS+post');
        else
            subplot(2,2,2); cla;
            subplot(2,2,3); cla;
            subplot(2,2,4); cla;
        end
    end

    function [scUsed, eyUsed] = getCalibCoords(scAll, eyAll, ...
            isUsed)
        % center is always the first coordinate
        scUsed = [CENTER scAll(:,isUsed)];
        eyUsed = [mean(eyAll(1:2,isUsed),2) eyAll(3:4,isUsed)];
        
    end

    function plotAllCalib(sc, ey, seld, good)
        % sc,ey,seld,good
        % sc is all points, in
        % u is logical w/ size = number of inputted points, (w/ center?)
        %badFixes = all(sc==0,1)';
        is_center = [true false(1,numel(seld)-1)];
        
        % build some vectors going from screen to eye to show resids:
        p = NaN(2, 3 * sum(seld));
        p(:, 1:3:3*sum(seld)) = sc(:,seld);
        p(:, 2:3:3*sum(seld)) = ey(:,seld);
        plot(p(1, :), p(2, :), 'b');
        hold on;
        
        plot([sc(1, 1) ey(1, 1)], ...
            [sc(2, 1) ey(2, 1)], 'r');
        
        % plot the screen coordinates themselves
        if(good(1))
            cen_style = 'r*';
        else
            cen_style = 'rx';
        end
        iszero = (sum(sc) == 0) ;
        plot(sc(1, 1), sc(2, 1), cen_style);
        
        plot(sc(1,~seld & ~iszero),sc(2,~seld & ~iszero), 'ro');
        
        plot(sc(1,seld & good & ~is_center), sc(2,seld & good & ~is_center), 'b*');
        plot(sc(1,seld & ~good & ~is_center), sc(2,seld & ~good & ~is_center), 'bx');
        hold off;
    end

    function [xi, yi, but] = getClick
        but = 0;
        while but ~= 1 % hitting return makes empty
            figure(gcf)
            [xi,yi,but] = ginput2(1);
            if isempty(but)
                break;
            end
        end
    end

    function d = dist(P1,P2)
        % p1 and p2 are 2x1
        d = sqrt((P1(1)-P2(1))^2+(P1(2)-P2(2))^2);
    end

    function ret = TPSisSet()
        ret = (~isempty(Calib.tpsInfo));
    end

end