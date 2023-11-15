% trainNPclassify2 %
% This is a function class called by for instance a simplex which is used
% to improve the workings of the NPclassify algorithm

function f = trainNPclassifyReal2(X)

% where to find files
dataSetPath1         = ['/home/nathan/aeroru/Objects_3.temp/Large/1/'];
dataSetPath4         = ['/home/nathan/aeroru/Objects_3.temp/Large/4/'];
dataSetPath8         = ['/home/nathan/aeroru/Objects_3.temp/Large/8/'];

% how to execute and log
execSetPath         = ['/home/nathan/aeroru/saliency/bin/'];
execFile            = ['test-NPclassify2'];
logFile             = [' >> ','/home/nathan/aeroru/saliency/matlab/trainNPclassifyState.',date,'.log'];
errOut              = ['2> /dev/null'];

% adjust sizing as needed

cd /home/nathan/aeroru/saliency/src;
unix(['rm -f ',execSetPath,'src/train_set.dat']);
fprintf('Listed, Running\n');
result = '';
resultComma = '';
doVarName = [];
doVarNameList = '';
command = '';
objects = [];

lsCommandData1       = ['ls -xm ',dataSetPath1, '*.ICA'];
lsCommandData4       = ['ls -xm ',dataSetPath4, '*.ICA'];
lsCommandData8       = ['ls -xm ',dataSetPath8, '*.ICA'];

[status,resultData1]     = unix(lsCommandData1);
[status,resultData4]     = unix(lsCommandData4);
[status,resultData8]     = unix(lsCommandData8);

% string before object number
resultData1s     = findstr(resultData1,['122']);
resultData4s     = findstr(resultData4,['122']);
resultData8s     = findstr(resultData8,['122']);

resultData1f     = findstr(resultData1,['ICA']);
resultData4f     = findstr(resultData4,['ICA']);
resultData8f     = findstr(resultData8,['ICA']);


fprintf('********************************************************\n');
fprintf('* NEW ROUND\n');
fprintf('********************************************************\n');
fprintf('* TRYING\n');
X

time = clock;

saveState = ['echo STATE ',date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),' ',...
            num2str(X(1)),' ',num2str(X(2)),' ',num2str(X(3)),' ',...
            num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),' ' ,num2str(X(7)),' X ',...
            num2str(X(8)),' ',num2str(X(9)),' ',num2str(X(10)),' ',num2str(X(11)),' ',num2str(X(11)),' ',...
            num2str(X(12)),' ',num2str(X(13)),' ',num2str(X(14)),' ',num2str(X(15)),' ',num2str(X(16)),' X ',...
            num2str(X(17)),' ',num2str(X(18)),' ',num2str(X(19)),' ',num2str(X(20)),' ',num2str(X(21)),' ',...
            num2str(X(22)),' ',num2str(X(23)),' ',num2str(X(24)),' ',num2str(X(25)),' ',num2str(X(26)),' X ',...
            num2str(X(27)),' ',num2str(X(28)),' ',num2str(X(29)),' ',num2str(X(30)),' ',num2str(X(31)),' ',...
            num2str(X(32)),' ',num2str(X(33)),' ',num2str(X(34)),' ',num2str(X(35)),' ',num2str(X(36)),' X ',...
            num2str(X(37)),' ',num2str(X(38)),' ',num2str(X(39)),' ',num2str(X(40)),' ',num2str(X(41)),' ',...
            num2str(X(41)),' ',logFile];

fprintf('STORING %s\n',saveState);
unix(saveState);
unix(['cd ',dataSetPath1]);
pwd
saveResults1 = zeros(size(resultData1s,1),size(resultData1s,2));
for i=1:size(resultData1s,2),
    doVarName       = resultData1(resultData1s(i):resultData1f(i)+2);
    fprintf('TRAINING[1] %d with %d objects\n',i,objects);
    command = [execSetPath,execFile,' ',dataSetPath1,doVarName,' ',...
               num2str(X(1)),' ',num2str(X(2)),' ',num2str(X(3)),' ',...
               num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),' ' ,num2str(X(7)),' ',...
               num2str(1),' ',num2str(1),' ',...
               num2str(X(8)),' ',num2str(X(9)),' ',num2str(X(10)),' ',num2str(X(11)),' ',num2str(X(11)),' ',...
               num2str(X(12)),' ',num2str(X(13)),' ',num2str(X(14)),' ',num2str(X(15)),' ',num2str(X(16)),' ',...
               num2str(X(17)),' ',num2str(X(18)),' ',num2str(X(19)),' ',num2str(X(20)),' ',num2str(X(21)),' ',...
               num2str(X(22)),' ',num2str(X(23)),' ',num2str(X(24)),' ',num2str(X(25)),' ',num2str(X(26)),' ',...
               num2str(X(27)),' ',num2str(X(28)),' ',num2str(X(29)),' ',num2str(X(30)),' ',num2str(X(31)),' ',...
               num2str(X(32)),' ',num2str(X(33)),' ',num2str(X(34)),' ',num2str(X(35)),' ',num2str(X(36)),' ',...
               num2str(X(37)),' ',num2str(X(38)),' ',num2str(X(39)),' ',num2str(X(40)),' ',num2str(X(41)),' ',...
               num2str(X(41)),' ',errOut];
    fprintf('command %s\n',command);
    [s,w] = unix(command);
    fprintf('result %s\n',w);
    saveResults1(i) = str2num(w);
end

unix(['cd ',dataSetPath4]);
pwd
saveResults4 = zeros(size(resultData4s,1),size(resultData4s,2));
for i=1:size(resultData4s,2),
    doVarName       = resultData4(resultData4s(i):resultData4f(i)+2);
    fprintf('TRAINING[4] %d with %d objects\n',i,objects);
    command = [execSetPath,execFile,' ',dataSetPath4,doVarName,' ',...
               num2str(X(1)),' ',num2str(X(2)),' ',num2str(X(3)),' ',...
               num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),' ' ,num2str(X(7)),' ',...
               num2str(1),' ',num2str(1),' ',...
               num2str(X(8)),' ',num2str(X(9)),' ',num2str(X(10)),' ',num2str(X(11)),' ',num2str(X(11)),' ',...
               num2str(X(12)),' ',num2str(X(13)),' ',num2str(X(14)),' ',num2str(X(15)),' ',num2str(X(16)),' ',...
               num2str(X(17)),' ',num2str(X(18)),' ',num2str(X(19)),' ',num2str(X(20)),' ',num2str(X(21)),' ',...
               num2str(X(22)),' ',num2str(X(23)),' ',num2str(X(24)),' ',num2str(X(25)),' ',num2str(X(26)),' ',...
               num2str(X(27)),' ',num2str(X(28)),' ',num2str(X(29)),' ',num2str(X(30)),' ',num2str(X(31)),' ',...
               num2str(X(32)),' ',num2str(X(33)),' ',num2str(X(34)),' ',num2str(X(35)),' ',num2str(X(36)),' ',...
               num2str(X(37)),' ',num2str(X(38)),' ',num2str(X(39)),' ',num2str(X(40)),' ',num2str(X(41)),' ',...
               num2str(X(41)),' ',errOut];
    fprintf('command %s\n',command);
    [s,w] = unix(command);
    fprintf('result %s\n',w);
    saveResults4(i) = str2num(w);
end

unix(['cd ',dataSetPath8]);
pwd
saveResults8 = zeros(size(resultData8s,1),size(resultData8s,2));
for i=1:size(resultData8s,2),
    doVarName       = resultData8(resultData8s(i):resultData8f(i)+2);
    fprintf('TRAINING[8] %d with %d objects\n',i,objects);
    command = [execSetPath,execFile,' ',dataSetPath8,doVarName,' ',...
               num2str(X(1)),' ',num2str(X(2)),' ',num2str(X(3)),' ',...
               num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),' ' ,num2str(X(7)),' ',...
               num2str(1),' ',num2str(1),' ',...
               num2str(X(8)),' ',num2str(X(9)),' ',num2str(X(10)),' ',num2str(X(11)),' ',num2str(X(11)),' ',...
               num2str(X(12)),' ',num2str(X(13)),' ',num2str(X(14)),' ',num2str(X(15)),' ',num2str(X(16)),' ',...
               num2str(X(17)),' ',num2str(X(18)),' ',num2str(X(19)),' ',num2str(X(20)),' ',num2str(X(21)),' ',...
               num2str(X(22)),' ',num2str(X(23)),' ',num2str(X(24)),' ',num2str(X(25)),' ',num2str(X(26)),' ',...
               num2str(X(27)),' ',num2str(X(28)),' ',num2str(X(29)),' ',num2str(X(30)),' ',num2str(X(31)),' ',...
               num2str(X(32)),' ',num2str(X(33)),' ',num2str(X(34)),' ',num2str(X(35)),' ',num2str(X(36)),' ',...
               num2str(X(37)),' ',num2str(X(38)),' ',num2str(X(39)),' ',num2str(X(40)),' ',num2str(X(41)),' ',...
               num2str(X(41)),' ',errOut];
    fprintf('command %s\n',command);
    [s,w] = unix(command);
    fprintf('result %s\n',w);
    saveResults8(i) = str2num(w);
end

% Sum error over all runs
E1 = (sum(saveResults1) - 1*size(resultData1s,2))
E4 = (sum(saveResults4) - 4*size(resultData4s,2))
E8 = (sum(saveResults8) - 8*size(resultData8s,2))
% Final error for all runs summed
EE = (E1+E4+E8)/(size(resultData1s,2)+size(resultData4s,2)+size(resultData8s,2));
fprintf('ERROR 1 %f\n',E1);
fprintf('ERROR 4 %f\n',E4);
fprintf('ERROR 8 %f\n',E8);
fprintf('FINAL %f\n',EE);

% write out results to a log file
time = clock;
saveState2 = ['echo ERROR ',date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
            ' ', num2str(EE),logFile];
unix(saveState2);

f = EE
