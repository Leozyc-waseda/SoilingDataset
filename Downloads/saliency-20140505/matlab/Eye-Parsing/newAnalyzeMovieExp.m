function CALIB = newAnalyzeMovieExp(ISCAN, PSYCH, scrw, scrh)
%function CALIB = analyzeMovieExp(ISCAN, PSYCH)

%scrw = 1920; scrh = 1080; % screen width and height


cenx = scrw/2-1; ceny = scrh/2-1; % screen coordinates of central fixation
didcal = 0; Mpre = eye(3); Mpost = eye(3);
sc = [cenx;ceny]; ey = [0;0]; % screen/eye position of center fixation cross
tpsc = 0.25;                  % coeff by which we apply the thin-plate-spline
cii = 1;                      % index of saved calibrated trace
outdist = 50;                 % outlier distance in pixels
minx = 2500; maxx = 2500; miny = 600; maxy = 600; % for test grid display
fishycal = 0;                 % will get true if last calibration was fishy

maxresi = 35;         % max acceptable residual distance, ppd is 35 pixel/degree for x, 33 pixel/degree for y
calerr = fopen('calerror.txt', 'w');

for ii = 1:length(PSYCH.fname)
  % get a data trace with regions where pupil bogus eliminated:
  data = cleanTrace(ISCAN.data{ii});
  disp(sprintf('\nanalyzing recording #: %i...', ii));

  % is it a calibration trace?
  if (strcmp(PSYCH.fname{ii}, 'CALIBRATION'))
    figure(3)
    plot([0 0 scrw-1 scrw-1 cenx PSYCH.fx(ii)], ...
	 [0 scrh-1 0 scrh-1 scrh-ceny scrh-PSYCH.fy(ii)], '*');
    axis([-10 scrw+10 -10 scrh+10]); grid on;
    figure(1)
    disp(['Parsing calibration for (' num2str(PSYCH.fx(ii)) ', ' ...
	  num2str(PSYCH.fy(ii)) ')...']); figure(1);
    [x1 y1 x2 y2] = parseCalibTrace(data);
    % if we got valid fixations, keep track of them:
    if (x1 ~= -1 & x2 ~= -1 & x1 ~= NaN & x2 ~= NaN & ...
       y1 ~= -1 & y2 ~= -1 & y1 ~= NaN & y2 ~= NaN) & ...
      x2 ~= 0 & y2 ~= 0
      % first element is the center point (for now the sum):
      ey(1, 1) = ey(1, 1) + x1;
      ey(2, 1) = ey(2, 1) + y1;
      % concatenate other locations if not center:
      if (PSYCH.fx(ii) ~= cenx | PSYCH.fy(ii) ~= ceny)
	sc = [sc [PSYCH.fx(ii); PSYCH.fy(ii)]]
	ey = [ey [x2; y2]]
      end
    end
    didcal = 0; % will need to do a calibration before next movie
  else % not a calibration trace
    % do we need to calibrate?
    if (didcal == 0)
      figure(1);
      % let's first normalize our center point:
      ey(1, 1) = ey(1, 1) / (size(ey, 2) - 1);
      ey(2, 1) = ey(2, 1) / (size(ey, 2) - 1);
        
      %let's find the most promising subset of these points
      [ey,sc] = findBestSubsetForAffineTrans(ey(1,:),ey(2,:),sc(1,:),sc(2,:),6,100000);
      
      % do a full calibration if this is our first time:
      if (~exist('tpsInfo') | fishycal)
	disp('*** Full calibration ***'); fishycal = 0;
	% Let's start with a pre-TPS affine calibration using all points:
	MpreOld = Mpre; % keep old calibration in case we abort
	Mpre = affineCalibration(sc, ey);
	eyc1 = calibAffine(ey, Mpre);
      
	% let's eliminate the outliers:
	idx = findOutliers(sc, eyc1, outdist);
	if (length(idx) < 5 & exist('tpsInfo'))
	  disp('not enough good points - keeping Mpre and TPS');
	  Mpre = MpreOld;
	  eyc1 = calibAffine(ey, Mpre);
	  eyc2 = deformPoints(eyc1', tpsInfo, [], tpsc)'; % warp
	  eyc = calibAffine(eyc2, Mpost);
	else
	  if (length(idx) < 6), fishycal = 1; end % this calib is fishy!
	  disp('Eliminating outliers in pre-Affine');
	  sc = sc(:, idx); ey = ey(:, idx);
	  Mpre = affineCalibration(sc, ey);
	  eyc1 = calibAffine(ey, Mpre);
	
	  % full TPS calibration:
	  % compute the TPS calibration between eyc and sc:
	  landmarks = [eyc1' sc'];  % target and template
	  tpsInfo = calcTPSInfo(landmarks);
	  
	  % get the deformed points for plotting:
	  eyc2 = deformPoints(eyc1', tpsInfo, [], tpsc)'; % warp
	  
	  % for now, Mpost is identity; it will come later if we have
	  % some post-calibration data:
	  Mpost = eye(3);
	  eyc = eyc2;
	end
      else
	% quick affine post-calibration: let's first get the points
        % transformed by the preAff and the TPS:
	disp('*** Recalibration ***');
	MpostOld = Mpost;
	eyc1 = calibAffine(ey, Mpre);
	eyc2 = deformPoints(eyc1', tpsInfo, [], tpsc)';
	Mpost = affineCalibration(sc, eyc2);
	eyc = calibAffine(eyc2, Mpost);

      	% let's eliminate the outliers:
	idx = findOutliers(sc, eyc, outdist*2);
	if (length(idx) < 4)
	  disp('not enough good points - keeping Mpost');
	  Mpost = MpostOld;
	  eyc = calibAffine(eyc2, Mpost);
	else
	  disp('Eliminating outliers in post-Affine');
	  sc = sc(:, idx); ey = ey(:, idx);
	  eyc1 = calibAffine(ey, Mpre);
	  eyc2 = deformPoints(eyc1', tpsInfo, [], tpsc)';
	  Mpost = affineCalibration(sc, eyc2);
	  eyc = calibAffine(eyc2, Mpost);
	end
      end
      
      % will not need to recalibrate unless we get new calibration data:
      didcal = 1;

      % show the residual distances:
      figure(2);
      subplot(2, 2, 1); plotCalib(sc, eyc1); title('after preAff');
      subplot(2, 2, 2); plotCalib(sc, eyc2); title('after TPS');
      subplot(2, 2, 3); plotCalib(sc, eyc); title('after postAff');

      resi1 = max(max(abs(sc - eyc)));
      resi2 = mean(mean(abs(sc - eyc)));
      disp(['max residuals is ' num2str(resi1) ', mean residuals is ' num2str(resi2)]);
      fprintf(calerr, '%f %f\n', resi1, resi2);
      if (resi1 > maxresi | resi2 > maxresi * 0.5)
	disp(['##### RESIDUALS TOO HIGH - RECOMMEND TRASHING DATA' ...
	      ' #####\g\g']);
      end
      
      % let's make a test grid and transform it:
      minx = min([minx ey(1, :)]); maxx = max([maxx ey(1, :)]);
      miny = min([miny ey(2, :)]); maxy = max([maxy ey(2, :)]);
      stepx = (maxx - minx)/30; stepy = (maxy - miny)/30;
      [eytx, eyty] = meshgrid(minx:stepx:maxx, miny:stepy:maxy);
      sizt = size(eytx); ss = sizt(1) * sizt(2);
      eyt = [reshape(eytx, 1, ss); reshape(eyty, 1, ss)];

      % pass the grid through the transforms:
      eytc1 = calibAffine(eyt, Mpre);         % pre-TPS affine transform
      eytc2 = deformPoints(eytc1', tpsInfo, [], tpsc)'; % thin-plate-spline
      eytc = calibAffine(eytc2, Mpost);       % post-TPS affine transform

      subplot(2, 2, 4); plot(eytc(1, :), eytc(2, :), '.');
      title('pre+TPS+post');
      figure(1);
    end
    figure(1); disp(['movie: ' PSYCH.fname{ii}]);

    % pass the trace through the full calibration business:
    ey = data(1:2, :);
    eyc1 = calibAffine(ey, Mpre);         % pre-TPS affine transform
    eyc2 = deformPoints(eyc1', tpsInfo, [], tpsc)'; % thin-plate-spline warping
    eyc = calibAffine(eyc2, Mpost);       % post-TPS affine transform
    caldata = data; caldata(1:2, :) = eyc;

    % do some final cleanup:
    caldata = removeTransients(caldata);

    % plot it:
    plotTrace(caldata, '-', 1);
    in = input('Store this data trace? [RET=yes]', 's');
    if (length(in) == 0 | (in(1) == 'y' | in(1) == 'Y'))
      % let's store the calibrated data:
      CALIB.fname{cii} = PSYCH.fname{ii};
      CALIB.data{cii} = caldata;
      CALIB.t(cii) = PSYCH.t(ii);
      cii = cii + 1;
    end

    % reinitialize the calibration accumulators:
    sc = [cenx; ceny]; ey = [0; 0];
  end
end
fclose(calerr);
