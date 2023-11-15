% tuningCurvesEdge.m
%
% script to plot tuning curves of the Reichardt Pyramid
% using an edge stimulus
%

% a couple of settings
stimSize = [20 3000];
numFrames = 11;
stepSizes = [1:50];
numSteps = size(stepSizes,2);
depth = 9;

plot_width = 3;
plot_height = 3;

maxMean = zeros(depth, numSteps);
maxStd = zeros(depth, numSteps);

avgMean = zeros(depth, numSteps);
avgStd = zeros(depth, numSteps);

panel = zeros(stimSize(1),stimSize(2)*2);
panel(:,stimSize(2)) = 1;

% loop over all speeds
for step=stepSizes
  
  % create stimuli
  img = zeros(numFrames,stimSize(1),stimSize(2));
  for frame=1:numFrames 
    left = stimSize(2)/2 + (frame - (numFrames + 1) / 2) * step;
    right = left + stimSize(2) - 1;
    img(frame,:,:) = panel(:,left:right);
  end
  
  % run the test
  [av,mn,mx] = TestReichardt(img,[1 0],depth);
  
  % store results
  avgMean(:,step) = mean(av(:,2:numFrames),2);
  avgStd(:,step) = std(av(:,2:numFrames),1,2);
  maxMean(:,step) = mean(mx(:,2:numFrames),2);
  maxStd(:,step) = std(mx(:,2:numFrames),1,2);
end

% now loop over the levels in order to plot the tuning curves
figure;
for lev=1:depth
  avgData = avgMean(lev,:);
  avgErr  = avgStd(lev,:);
  
  maxNorm = max(maxMean(lev,:),[],2);
  if (maxNorm == 0)
    maxNorm = 1;
  end
  maxData = maxMean(lev,:) / maxNorm;
  maxErr  = maxStd(lev,:) / maxNorm;
  
  subplot(plot_height,plot_width,lev);
  %errorbar(stepSizes',maxData,maxErr,'r--');
  %hold on;
  errorbar(stepSizes',avgData,avgErr,'b-');
  %hold off;
  title(sprintf('Level: %i',lev-1));
end

