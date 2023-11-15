function Cdata = analyzeMovieExpInter(ISCAN, PSYCH, scrw, scrh, do_driftcorr)
% function CALIBdata = analyzeMovieExp(ISCAN, PSYCH)
% Main program for converting ISCAN coordinates to session coordinates
% NB: no cleaning is done to the data, but only provisionally to
% run calibration and for visualization

CENX = scrw/2-1; CENY = scrh/2-1; % screen coordinates of central fixation

if(exist(backupFile(1),'file')) %if the first backup file exists
    in = input('Previous run crashed - restore from previous? [RET=yes]', 's');
    if (~isempty(in) && lower(in(1)) ~= 'y')
        clearBackups;
    end
end

% find all trials that are calibration
calibtrials = strmatch('CALIBRATION',PSYCH.fname);
df = diff(calibtrials);
% matlab golf code to find beginning and end of calib blocks
calibblocks = calibtrials([find([Inf;df]>1), find([df;Inf]>1)]);
ncalibs = size(calibblocks, 1);
ntrials = length(PSYCH.fname);

%set up iteration
ii = 1;
cii = 1;                      % index of saved calibrated trace
needcal = 1;
fishycal = 0;                 % will get true if last calibration was fishy
sc = [CENX;CENY]; ey = [0;0]; % screen/eye position of center fixation cross
% TODO: autodetect N-pt calibs: preset only valid for 9-pt calibration
nCalibPts = 9;

% run drift correction?
if nargin < 5
    do_driftcorr = false;
end

% iterate over recordings, increment at end
while (ii <= ntrials)
    
    fprintf(1, '\nanalyzing recording #: %i...\n', ii);
    
    % get a data trace with regions where pupil bogus eliminated:
    dirtydata = ISCAN.data{ii};
    data = cleanTrace(dirtydata);
    
    % is it a calibration trace?
    if ismember(ii,calibtrials)
        if (ii == 1) || ~strcmp('CALIBRATION',PSYCH.fname(ii-1))
            firstCalib = ii;
            
            %initialize variables
            fixResults = zeros(4,nCalibPts);
            useCalib = false(nCalibPts,1);
            screenCoords = zeros(2,nCalibPts);
        end
        reli = ii-firstCalib+1;
        
        % plotting setup - plot saccade target and monitor borders
        % this is the true position, which is why y is inverted
        figure(3)
        plot([CENX PSYCH.fx(ii)], [scrh-CENY scrh-PSYCH.fy(ii)], 'b*', ...
            [0 0 scrw-1 scrw-1 0], [0 scrh-1 scrh-1 0 0], 'r-', 'LineWidth', 2) ;
        axis([-10 scrw+10 -10 scrh+10]); grid on;
        % FIXME: should flip y-axis labels to reflect screen geometry
        
        figure(1);
        
        x1=-1; x2=-1; y1=-1; y2=-1;
        % use backup if it exists
        if(exist(backupFile(ii),'file')) % if file exists use it
            fprintf(1, 'Loading previous fixation for trial %d\n', ii);
            load(backupFile(ii),'x1', 'x2', 'y1', 'y2');
        else
            % get the fixations from each calibration trace
            
            fprintf(1, 'Parsing calibration for (%d, %d)...\n', PSYCH.fx(ii), PSYCH.fy(ii));
            if PSYCH.fx(ii) == CENX && PSYCH.fy(ii) == CENY % don't process the center
                disp('Skipping center point for processing');
            else 
                [x1 y1 x2 y2] = parseCalibTraceInter(cleanTrace(data));
            end
            save(backupFile(ii),'x1', 'x2', 'y1', 'y2');
        end
        fixResults(:,reli) = [x1 y1 x2 y2]';
        
        % if we got valid fixations, keep track of them:
        if (~ismember(-1,[x1 x2 y1 y2])) && ~any(isnan([x1 x2 y1 y2]))
            % first element is the center point (for now the sum):
            ey(1:2,1) = ey(1:2,1) + [x1 y1]';
            
            % concatenate other locations if not center:
            if (PSYCH.fx(ii) ~= CENX || PSYCH.fy(ii) ~= CENY)
                sc = [sc [PSYCH.fx(ii); PSYCH.fy(ii)]];
                screenCoords(:,reli) = [PSYCH.fx(ii); PSYCH.fy(ii)];
                ey = [ey [x2; y2]];
                %eyC(reli,:) = [x2; y2];
            end
            useCalib(reli) = true;
            
        else % fixation is bad
            disp('Fixation not found... skipping');
        end
        needcal = 1; % will need to do a calibration before next movie
        
    else % not a calibration trace
        % do we need to calibrate?
        if (needcal)          
            % let's first normalize our center point:
            ey(1:2, 1) = ey(1:2, 1) / (size(ey, 2) - 1);
            
            % do a full calibration if this is our first time or
            % calibration is fishy:
            
            % TODO: maybe try selection of points
            % interactively over keyboard?
            % FIXME: this try catch block gets upset at versions <= R2007a
            figure(2)
            try
                if (~exist('calib','var') || fishycal)
                    fishycal = 0;
                    fprintf(1,'*** Full calibration (n = %i) ***\n',size(ey,2) - 1);
                    calib = findFullCalibInter(screenCoords, fixResults,...
                        useCalib, [CENX; CENY]); %fresh new calibration
                else
                    % quick affine post-calibration: let's first get the points
                    % transformed by the preAff and the TPS:
                    fprintf('*** Recalibration (n = %i) ***\n',size(ey,2) - 1);
                    calib = findReCalib(sc, ey, calib);
                end
            catch ME
                fishycal = 1;
                
                %re-report message
                fprintf(1,'%s: %s\n', ME.identifier, ME.message);
                i_c = find(calibblocks(:,2)==ii-1);
                
                % clear all the bad CALIB backups
                clearBackups(calibblocks(i_c,1):calibblocks(i_c,2));
                
                sc = [CENX; CENY]; ey = [0; 0]; % reset accumulators
                
                % go back to first calibration trace and try again
                in = input('Previous calibration failed: re-try? [RET=yes]', 's');
                
                if (isempty(in) || lower(in(1)) == 'y')
                    % skip the increment and go back to the first calib
                    ii = calibblocks(i_c,1);
                    continue;
                end
                
                % force calibration?
                in = input(['Force calibration to apply (not recommended) ?'...
                    '[RET=no]'], 's');
                if isempty(in) || lower(in(1)) ~= 'y' % do not force
                    
                    % option to skip data
                    in = input(['Skip this set of data and ' ...
                        'move to next calibration? [RET=no, and quits]'], 's');
                    if ~isempty(in) && lower(in(1)) == 'y'
                        if i_c == ncalibs, ii = ntrials+1;
                        else ii = calibblocks(i_c+1,1);
                        end
                        continue;
                    end
                    
                    % exit with error
                    fprintf('Exiting...\n');
                    rethrow(ME);
                end % if forcing
            end %try catch
            
            
            % will not need to recalibrate unless we get new calibration data:
            needcal = 0;
        end %if(needcal)
        
        figure(1); disp(['stimulus: ' PSYCH.fname{ii}]);
        
        % use backup if it exists
        if(exist(backupFile(ii),'file')) % if file exists use it
            fprintf(1, 'Loading previous trace for trial %d\n', ii);
            load(backupFile(ii),'record');
            
            % save the data in memory
            Cdata(cii) = record;
            cii = cii + 1;
        else
            
            % pass the cleaned trace through the full calibration business:
            ey = data(1:2, :);
            caldata = data;
            caldata(1:2, :) = calibFull(ey, calib);
            
            % at this point, check drift correction
            % this is zeroth order - global shift on Mpost
            initFix = caldata(1:2, 1);
            FIX_TOL = 50;
	    FIX_NOTON = 150;

	    % TODO: check previous fixation's drift correct status
            if do_driftcorr && ...
		  any(initFix-[CENX;CENY] > FIX_TOL) && ...
		  all(initFix-[CENX;CENY] < FIX_NOTON) 
                fprintf(['WARNING: Initial fixation is off center,\n'...
                        'applying 0th order DRIFT CORRECTION...\n']);
                    calib.Mpost = shiftAffine(calib.Mpost, initFix-[CENX;CENY]);
            end
            % plot the data:
            plotTrace(caldata, 'm-', 1);
            
            % print basic statistics
            len = length(caldata);
            perblank = sum(any(isnan(data),1))/len;
            fprintf('Trace is %d samples, %2.1f%% blanks.\n', len, perblank*100);
            
            % load data into memory
            treject = 0.9;
            if (perblank>treject)
                in = input('Store this data trace? [RET=no]', 's');
            else
                in = input('Store this data trace? [RET=yes]', 's');
            end
            
            if (isempty(in) && perblank<treject || ...
                    ~isempty(in) && lower(in(1)) == 'y')
                % let's store the calibrated data:
                
                record.fname = PSYCH.fname{ii};
                % all cleanup should be done in eye-markup
                record.data = dirtydata;
                record.data(1:2,:) = calibFull(dirtydata(1:2,:),calib);
                record.t = PSYCH.t(ii);
                if isfield(ISCAN,'hz')
                    record.hz = ISCAN.hz(ii); %carry sampling information
                else
                    record.hz = 240; %TODO: remove hardcoded sf default
                end
                
                % save to memory
                Cdata(cii) = record;
                cii = cii + 1;
                
                % and keep track of it on file
                save(backupFile(ii),'record');
            end
        end
        
        % reinitialize the calibration accumulators:
        sc = [CENX; CENY]; ey = [0; 0];
    end
    
    ii = ii + 1; %end while loop for recordings
end

disp('Clearing backups...');
clearBackups;
