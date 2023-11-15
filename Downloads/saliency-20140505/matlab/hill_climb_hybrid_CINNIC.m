%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hill_climb_hybrid_CINNIC.m                       %
%                                                                                                  %
% T. Nathan Mundhenk nathan@mundhenk.com           %
%                                                                                                  %
% Computes for hill climbing as many variables as  %
% there are nodes on the beowulf cluster. For each %
% step, the best step is taken if it improves the  %
% state of the system. If that cannot be done, go  %
% backwords on the worst. Failing both, move in    %
% a random direction.                                                      %
%                                                                                                  %
% Developed under matlab 5.3                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

maxIter = 20;           % maximum number of iterations
cardinal = [];          % cardinal set of all return values
cardRatio = [];     % ratio of base line to new measure
cardLog = [];           % the log threshold elevation
%stims = [0, -0.2, 35, 60, 1.1, 101, 2, 30, 8, 10, 50000, 1.0001, 0.95, 0.9, 0.65, 400];      % 1st train set stims, funky because shrunk kernel
%stims = [0,0,35,60,1.1,100,2,30,12,10,50000,1.0001,1,0.9,0.7,400];             % initial stim size, will hold further values
%stims = [0.1, 0, 36, 60, 1, 100, 2, 30, 5, 12, 50000, 1.0001, 0.45, 0.85, 0.7, 400];      % Best error so far, no kernal shrink

%stims = [0,0,0,0, 0,0,0,0, 35,60,1.1,100 ,0.95,0.9,0.65,400];

%stims = [0,0,-0.2,-0.1, 0,0,0,0, 36,60,1,100, 0.45,0.9,0.75,420];
%stims = [0,0.1,-0.1,-0.1, 0,0,0,0, 36,59,1,100, 0.45,0.9,0.7,420];
%stims = [-0.05,-0.1,0,0, 0,-0.05,0,0.05, 33,61,0.9,100, 0.6,0.95,0.7,400];
%stims = [-0.1,-0.05,0.05,0, 0.05,-0.1,-0.05,0, 32,60,0.9,100, 0.45,0.9,0.75,410]
%stims = [-0.075,-0.075,0.05,0, 0.05,-0.1,-0.075,0, 31,60,0.85,96, 0.45,0.9,0.75,410];
stims = [-0.075,-0.075,0.05,-0.01, 0.05,-0.1,-0.075,0, 31,60,0.85,96, 0.45,0.9,0.77,410];
%stimsLB = [-50,-50,15,15,0.5,25,1,1,3,8,5000,1.00001,0,0,0,25];                % stims lower bound
stimsLB = [-50,-50,-50,-50, -50,-50,-50,-50, 15,15,0.1,1, 0.1,0.1,0.1,25];
%stimsUB = [50,50,70,70,4,1000,2,1000,8,12,150000,2,1,1,1,1500];                % stims upper bound
stimsUB = [50,50,50,50, 50,50,50,50, 80,80,4,500, 1,1,1,1500];

%stimsType = [2,2,1,1,2,2,1,2,2,1,1,2,2,2,2,1]; % 0 - bool, 1 - discrete, 2 - continious
stimsType = [2,2,2,2, 2,2,2,2, 1,1,2,1, 2,2,2,1];
stimsDisc = [];         % tells the minimum discrete step of a variable
%stimsStep = [0.1,0.1,2,2,.05,1,1,1,2,2,5000,0.0001,0.05,0.05,0.05,20];         % ideal step size for a stim

%stimsStep = [0.01,0.01,0.01,0.01, 0.01,0.01,0.01,0.01, 1,1,0.05,1, 0.02,0.02,0.02,2]; % tiny steps
stimsStep = [0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1, 2,2,0.1,2, 0.05,0.05,0.05,10]; %medium steps
%stimsStep = [0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5, 10,10,.5,20, 5,5,5,50]; % BIG STEPS
hardConst = [1,1,-1,-1,-1,-1,0,0];      % hard constraints on results -1,+1 if don't care 0
lastbest = 0;           % last item that yeilded best results
lastbestVal = 0;        % last value that yeilded best results
randMax = 3;            % max number of random guesses before quiting

% last type of step taken
%0 - random
%1 - forward on best
%2 - backward on worst
stepType = -1;

stepDirection = -1; % last item to be changed, ie step direction
randCount = 0; % count how many times a random step was taken in a row. Quit on max.
stimsStepDir = []; % stores direction of stim steps -1 for backwards, +1 for forwards
errorDir = []; % direction of error. Determines whether error got better or worse for a var
errorDirMean = [];
errorDirStd = [];
lastmeanError = 0; % set this up so matlab won't complane
% for maxIter iterations do this. Since state is preserved, this number need not be large.

fprintf('\n***********************************************************\n');
fprintf('* START\n');
fprintf('***********************************************************\n');

for t=0:maxIter,

    fprintf('\n>>>>> ITERATION %d\n\n',t);

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % COMPUTE STEP and STEP DIRECTION                  %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % for first iteration take a random step direction %

   if t == 0,
      for i=1:size(stims,2),
         if stimsType(i) ~= 0,
                if rand(1) > .5,
                                stimsStepDir(i) = -1;
                fprintf('MATLAB: %d taking step %d\n',i,stimsStepDir(i));
                else
                stimsStepDir(i) = 1;
                fprintf('MATLAB: %d taking step %d\n',i,stimsStepDir(i));
                end
         else
            if stims(i) == 0,
               stimsStepDir(i) == 1;
               fprintf('MATLAB: %d taking step %d\n',i,stimsStepDir(i));
            else
               stimsStepDir(i) == -1;
               fprintf('MATLAB: %d taking step %d\n',i,stimsStepDir(i));
            end
         end
      end
   end

   % for iteration of t = 1 just keep going the same way for error comp %

   % if iteration is greater then 0, use simplex like steps for all vars %
   % start this at t > 1 since two iterations are needed to make compairisons %
   % errorDir - direction of error, increase v. decrease
   % fail - failed hard constaint on value
   % stepType - if 2, all values must try a step backwards
   if t > 1,
      errorDir = meanError - lastmeanError;
      errorDirSum(:,t) = errorDir;
      % determine step directions %
      for i=1:size(stims,2),
         errorDirMean(i) = mean(errorDirSum(i,:));
         errorDirStd(i) = std(errorDirSum(i,:));
         if stimsType(i) ~= 0,
            % error stayed the same, random step %
            if errorDir(i) == 0 & fail(i) == 0 & stepType ~= 2,
                if rand(1) > .5,
                                        stimsStepDir(i) = stimsStepDir(i) * (-1);
                    fprintf('MATLAB: %d taking random step OTHER way %d (errorDir %d fail %d)\n',i,stimsStepDir(i),errorDir(i),fail(i));
                else
                    fprintf('MATLAB: %d taking random step SAME way %d (errorDir %d fail %d)\n',i,stimsStepDir(i),errorDir(i),fail(i));
                end

            end
            % error increased step back, reverse step %
            if errorDir(i) > 0 | fail(i) == 1 | stepType == 2,
                   stimsStepDir(i) = stimsStepDir(i) * (-1);
                   fprintf('MATLAB: %d taking step BACK %d (errorDir %d fail %d)\n',i,stimsStepDir(i),errorDir(i),fail(i));
            end
            % error decreased - DEFAULT, forward step %
            if errorDir(i) < 0 & fail(i) == 0 & stepType ~= 2,
                fprintf('MATLAB: %d taking step FORWARD %d (errorDir %d fail %d)\n',i,stimsStepDir(i),errorDir(i),fail(i));
            end
         else
            % simply toggle for i %
            % ignore if it got better %
            if errorDir(i) < 0
                if stimsType(i) == 0,
                    if stimsStepDir(i) == 0,
                        stimsStepDir(i) == 1;
                    else,
                            stimsStepDir(i) = -1;
                    end
                end
                fprintf('MATLAB: %d TOGGLE step %d\n',i,stimsStepDir(i));
            else
                fprintf('MATLAB: %d NON-TOGGLE step %d\n',i,stimsStepDir(i));
            end
         end
      end
      errorDirMean
      errorDirStd
   end


   stimsCurrent = stims + (stimsStepDir .* stimsStep);

   % stay inside bounds
   % if over bounds, set to bound, then ask to reverse direction next time

   for i=1:size(stims,2),
        if stimsCurrent(i) < stimsLB(i),
            fprintf('MATLAB: %d exceded lower bound\n',i);
                stimsCurrent(i) = stimsLB(i);
            stimsStepDir(i) = stimsStepDir(i) * (-1);
      end
      if stimsCurrent(i) > stimsUB(i),
            stimsCurrent(i) = stimsUB(i);
            fprintf('MATLAB: %d exceded upper bound\n',i);
            stimsStepDir(i) = stimsStepDir(i) * (-1);
      end
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % RUN PERL for CINNIC                                                  %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        OS = '';

        for i=1:size(stims,2),
            OS = [OS,' ',num2str(stimsCurrent(i))];
        end
        for i=1:size(stims,2),
        OS = [OS,' ',num2str(stims(i))];
    end

   % run CINNIC polat sagi based tests in perl

   fprintf('VARIABLES %s\n',OS);

   cd /lab/mundhenk/code/saliency/src;
   callCommand = ['perl ../perl/doPolatSagiSimplex.pl ',OS];
   %commandKill = unix('killall -9 perl','-echo');
   commandResult = unix(callCommand,'-echo');

   fprintf('RUNNING STATS\n');

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % COLLECT AND ANALIZE RESULTS                      %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   cardBase = [];
   cardinal = [];
   for i=1:size(stims,2),
        loadThis = ['../PolatSagi/',num2str(i-1),'/finalMatlab.',num2str(i-1),'.dat'];
                MAT = load(loadThis);
            Major = MAT(:,3:size(MAT,2)); Minor = MAT(:,1);
            Base = MAT(:,2);
            siz(i) = size(MAT,1); cardinal = cat(1,cardinal,Major); cardBase = cat(1,cardBase,Base);
   end

   load('rawDataSet.dat');

   % determine error for data

        for i=1:size(cardinal,1),
            for j=1:size(cardinal,2),
            cardRatio(i,j) = cardinal(i,j)/cardBase(i);
            end
        end

   % find log scale for threshold elevation

    cardLog = log10(cardRatio);

   % subtract found threshold from known threshold

   rawError = zeros(size(cardLog,1),size(cardLog,2));

   for i=1:size(cardLog,1),
        for j=1:size(cardLog,2),
            rawError(i,j) = cardLog(i,j) - rawDataSet(j);
        end
    end

   rawDataSet
   cardLog
   rawError

   totalError = zeros(1,size(cardinal,1));

   totalError = sum(abs(rawError),2);

   % find total error under each condition
   % also find mean error for each condition

   % store last mean error for future use %
   if t > 0,
        lastmeanError = meanError;
   end

   meanError = totalError ./ size(cardLog,2);

   meanError

   % find the total mean error for the whole set

   meanErrorAll = sum(totalError)/size(cardLog,1);

   % check for bounding in hard constraints
   % a value should be possitive or negative in certain cercumstances

   fail = zeros(size(cardLog,1),1);

   for i=1:size(cardLog,2),
      Check = zeros(1,size(cardLog,1));
      for j=1:size(cardLog,1),
         Check(j) = cardLog(j,i) * hardConst(i);
         if Check(j) < 0,
            fail(j) = 1;
         end
      end
   end

   % find lowest error condition based upon total error and
   % hard constraint violation

   best = -1;
   bestVal = 0;
   worst = -1;
   worstVal = 0;

   for i=1:size(meanError,1),
      % determine best improvement
      if best == -1,
         if fail(i) ~= 1,
            best = i;
            bestVal = meanError(i);
            fprintf('MATLAB: Found first best %d\n',i);
         end
      else,
         if fail(i) ~= 1,
            if meanError(i) < bestVal,
               best = i;
               bestVal = meanError(i);
               fprintf('MATLAB: Found subsequent best %d\n',i);
            end
         end
      end
      % determine worst improvement
      if worst == -1,
         worst = i;
         worstVal = meanError(i);
         fprintf('MATLAB: Found first worst %d\n',i);
      else,
         if meanError(i) > worstVal,
            worst = i;
            worstVal = meanError(i);
            fprintf('MATLAB: Found subsequent worst %d\n',i);
         end
      end
   end

   % how much better or worse is the best/worst compaired with the mean

   worstDiff = abs(worstVal - meanErrorAll);
   bestDiff = abs(bestVal - meanErrorAll);

   % find out if we got better this time or worse %
   % if better, set lastbestval %

   gotworse = 0;
   if t > 0,
        if bestVal >= lastbestVal | best == -1,
            fprintf('MATLAB: EXCEPTION - bestVal >= lastbestVal or No bestVal\n');
            gotworse = 1;
        else
            lastbestVal = bestVal;
        end
    else,
       lastbestVal = bestVal;
   end


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % DECIDE WHAT KIND OF STEP TO TAKE NEXT            %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % decide which values to augment based upon new findings
   quit = 0;
   move = -1;
   doVar = 666;
   randFound = 0;

   % first iteration steps %
   if stepType == -1,
                if gotworse == 0,
         % take best step %
         move = 0;
         doVar = best;
         stepType = 1;
         fprintf('MATLAB: 1 Taking Best Step %d\n',doVar);
            else,
         % backtrack, inverse worst %
         move = 1;
         doVar = worst;
         stepType = 3;
         fprintf('MATLAB: 1 Reversing On Worse Step %d\n',doVar);
       end
   else
   % second iteration steps %
   % decide which type of step to take
   % STEP 0. Take the best step always %
        if gotworse == 0,
         % take best %
         move = 0;
         doVar = best;
         stepType = 1;
         % reset random counter %
         randCount = 0;
         fprintf('MATLAB: 2.a Taking Best Step %d\n',doVar);
         if t > 1,
            fprintf('\tImprovement: %f Error: %f',errorDir(doVar),meanError(doVar));
         else
            fprintf('\tImprovement: Error: %f',meanError(doVar));
         end
    % STEP 1-4, if it doesn't get better, try and cope %
        else,
        % STEP 4. Can't get better, quit %
         if stepType == 0,
            if randCount == randMax,
               t = maxIter;
               fprintf('MATLAB: 2.e GIVING UP! cannot improve error, took %d guesses\n', randCount);
            else
                stepType = 3;
            end
        end
        % STEP 1. no improvement after one step, everyone try the reverse %
        if stepType == 1,
            stepType = 2;
            quit = 1;
            move = 2;
            fprintf('MATLAB: 2.b Reversing: no doVar, no perm step\n');
        end
        % STEP 2. backtrack, inverse worst %
        if stepType == 2 & quit == 0,
            move = 1;
            doVar = worst;
            stepType = 3;
            quit = 1;
            fprintf('MATLAB: 2.c Reversing On Worse Step %d\n',doVar);
            fprintf('\tImprovement: %f Error: %f',errorDir(doVar),meanError(doVar));
         end
         % STEP 3. Back track didn't work, guess with a random step type %
         if stepType == 3,
            if quit == 0 & randFound == 0,
               moveTemp = []; doVarTemp = [];
                   moveTemp = round(rand(1));
               doVarTemp = round(rand(1) * (size(stims)-1));
               move = moveTemp(2) + 1;
               doVar = doVarTemp(2) + 1;
               stepType = 0;
               % incremement random counter %
               randCount = randCount + 1;
               randFound = 1;
               fprintf('MATLAB: 2.d Taking a Random Step %d Number %d\n',doVar,randCount);
            end
        end
      end
   end

   % compute next step %

   doVar

   if move == 0,
      % forward computation step %
      stims(doVar) = stimsCurrent(doVar);
   end
   if move == 1,
      % backward computation step %
      % make sure this isn't a bool %
      if stimsType(doVar) ~= 0,
         stims(doVar) = stims(doVar) - (stimsCurrent(doVar) - stims(doVar));
         % check bounds %
         if stims(doVar) < stimsLB(doVar),
            stims(doVar) = stimsLB(doVar)
         end
         if stims(doVar) > stimsUB(doVar),
            stims(doVar) = stimsUB(doVar)
         end
      else
         % if a bool, just toggle it %
         if stimsCurrent(doVar) == 1,
            stims(doVar) = 0;
         else
            stims(doVar) = 1;
         end
      end
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % LOG STATE OF SYSTEM                              %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   OS = '';
   meanErrorLog = '';
   failLog = '';
   % log state %
   for i=1:size(stims,2),
      OS = [OS,' ',num2str(stims(i))];
      meanErrorLog = [meanErrorLog,' ',num2str(meanError(i,1))];
      failLog = [failLog,' ',num2str(fail(i,1))];
   end

   OS

   doThis = ['echo ',date,' ',time,' ',num2str(t),' ',num2str(doVar),' ',OS,' >> stims_result.log'];
   unix(doThis,'-echo');
   if doVar > 0 & doVar < 666,
        doThis = ['echo ',date,' ',time,' ',num2str(t),' VARIABLE ',num2str(doVar),' TYPE ',num2str(stepType),' DIRECTION ',num2str(move),' ERROR ', num2str(meanError(doVar,1)),' >> stims_go.log'];
   else,
        doThis = ['echo ',date,' ',time,' ',num2str(t),' VARIABLE ',num2str(doVar),' TYPE ',num2str(stepType),' DIRECTION ',num2str(move),' ERROR 000 >> stims_go.log'];
   end
   unix(doThis,'-echo');
   doThis = ['echo ',date,' ',time,' ',num2str(t),' ',num2str(doVar),' ',meanErrorLog,'>> meanError.log'];
   unix(doThis,'-echo');
   doThis = ['echo ',date,' ',time,' ',num2str(t),' ',num2str(doVar),' ',failLog,'>> fail.log'];
   unix(doThis,'-echo');
end



