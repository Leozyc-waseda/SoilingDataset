% trainNPclassify2 %
% This is a function class called by for instance a simplex which is used
% to improve the workings of the NPclassify algorithm

%function f = trainNPclassifyReal(X)
%X = [2;5.949;2.3447;-0.77114;9.1723;27.6191;-0.0024981;1.415;0.74183];
%X =  [-0.1;5.949;0.68604;32.2476;-10.7899;69.0802;-0.0024981;1.415;0.74183];
%X =  [1.9707;5.9485;2.6479;4.1252;9.863;57.9016;0.048849;1.0329;0.80935];
%X = [2;5.8417;2.5702;2.0786;9.6807;28.8574;-0.0098149;1.6478;0.67709];
%X = [2 6.0959 2.5 3 13.0363 2.0 -0.051474 0.48111 -0.29432];
%clear
%baseX = [ 1.8333 5.9524 2.5 3 4.899 2.8148 1];
%X = [2.7516 6.1991 2.5 3 4.899 2.8148 0.77809];

%X = [2.7516 6.1991 2.5 3 9.899 2.8148 7]; % objects
%X = [3.0603 6.1991 2.5 3 9.899 2.7148 6.5722];

%X = [2.0722 5.9524 2.5 3 15.5599 2.8148 0]; % real scenes
X = [3.4271 5.9524 2.5 3 15.7383 2.7148 0.46677];
%baseChildIC = -0.20775;
%baseChild = 9.6807;
%baseSize = 29.1816;



% where to find files
setSeries           = ['nat'];
dataSetPath         = ['/lab/mundhenk/tmp/data_set/',setSeries,'/'];
winnerSetPath       = ['/lab/mundhenk/tmp/winner/',setSeries,'/'];
imageSetPath        = ['/lab/mundhenk/tmp/image_3/'];

% how to execute and log
execSetPath         = ['/lab/mundhenk/code/saliency/'];
execFile            = ['test-NPclassify2'];
logFile             = [' >> ','trainNPclassify.',date,'.log'];

% 1 if you want to get output images, 0 if not
doImage             = 1;


%basePoly1 = -0.051474;
%basePoly2 = 0.48111;
%basePoly3 = -0.19432;

% adjust sizing as needed
%X(1:6) = ((X(1:6) - baseX) * 10) + X(1:6);
%X(2) = ((X(2) - baseX(2)) * 10) + X(2);
%X(3) = ((X(3) - baseX(3)) * 10) + X(3);
%X(7) = ((X(7) - basePoly1) * -0.1) + X(7);
%X(8) = ((X(8) - basePoly2) * .1) + X(8);
%X(9) = ((X(9) - basePoly3) * -0.1) + X(9);

cd /lab/mundhenk/code/saliency/src;
unix(['rm -f ',execSetPath,'src/train_set.dat']);
fprintf('Listed, Running\n');
result = '';
resultComma = '';
doVarName = [];
doVarNameList = '';
command = '';
objects = [];

lsCommandData       = ['ls -xm ',dataSetPath];
lsCommandWinner     = ['ls -xm ',winnerSetPath];
lsCommandImage      = ['ls -xm ',imageSetPath];

[status,resultData]     = unix(lsCommandData);
[status,resultWinner]   = unix(lsCommandWinner);
[status,resultImage]    = unix(lsCommandImage);

% string before object number
resultDataO     = findstr(resultData,['.300.']);
resultWinnerO   = findstr(resultWinner,['.300.']);
% string between which the file name can be found
resultDataM     = findstr(resultData,'ICA');
resultWinnerM   = findstr(resultWinner,'nat');
resultImageM    = findstr(resultImage,'nat');

resultDataN     = findstr(resultData,'dat');
resultWinnerN   = findstr(resultWinner, 'dat');
resultImageN    = findstr(resultImage, 'ppm');


fprintf('********************************************************\n');
fprintf('* NEW ROUND\n');
fprintf('********************************************************\n');
fprintf('* TRYING\n');
X

time = clock;

saveState = ['echo STATE ',date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
            ' ',num2str(X(1)),' ',num2str(X(2)),' ',num2str(X(3)),' ',...
            num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),' ' ,num2str(X(7)),' ',num2str(1),' ',num2str(1),...
            logFile];
fprintf('STORING %s\n',saveState);
unix(saveState);

for i=1:size(resultDataM,2),
    doVarName       = resultData(resultDataM(i):resultDataN(i)+2)
    doWinnerName    = resultWinner(resultWinnerM(i):resultWinnerN(i)+2);
    doImageName     = resultImage(resultImageM(i):resultImageN(i)+2);
    objectIdeal(i)  = 1;%str2num(resultData((resultDataO(i)+5)));
    objects         = objectIdeal(i);

    fprintf('TRAINING %d with %d objects\n',i,objects);

    command = [execFile,' ',dataSetPath,doVarName,' ',num2str(objects),' ',num2str(X(1)),...
            ' ',num2str(X(2)),' ',num2str(X(3)),' ',num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),...
            ' ',num2str(1),' ',num2str(1),' ',num2str(1),' ',winnerSetPath,doWinnerName,...
            ' ',num2str(doImage),' ',imageSetPath,doImageName,' ',num2str(X(7))];
    fprintf('command %s\n\n',command);
    unix(command);
end

% load results from output file from executions
fprintf('Processing Error\n');
load('train_set.dat');
fprintf('Loaded error file\n');

% error from absolute value of error summed
ff = ((sum(abs(train_set(:,6) - transpose(objectIdeal))))/size(resultDataN,2)); %objects
% prior probability of error, e.g. target
CH = sum(objectIdeal)/size(resultDataO,2);
fprintf('ERROR %f\n',ff);
fprintf('PRIOR %f\n',CH);

% write out results to a log file
time = clock;
saveState2 = ['echo ERROR ',date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
            ' ', num2str(ff),logFile];
unix(saveState2);

f = ((sum(abs(train_set(:,6) - transpose(objectIdeal))))/size(resultDataN,2));
