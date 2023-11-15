%
% train-NPclassify.m %
% 1. take output suggestions from NPclassify along with statistics on the space
% 2. fit data with perhaps a regression method
% 3. test training
cd /lab/mundhenk/code/CODE_BASE_2/saliency/src;
unix('rm -f /lab/mundhenk/code/CODE_BASE_2/saliency/src/train_set.dat');
result = '';
resultComma = '';
doVarName = [];
doVarNameList = '';
command = '';
Fit = []; Val = []; Err = []; INPl = []; INP = []; OUT = []; objects = [];
[status,result] = unix('ls -xm ../data_sets/num/');
resultO = findstr(result,'o');
resultM = findstr(result,'m');
resultN = findstr(result,'j');
trainList = zeros(size(resultO));
objectIdeal = zeros(size(resultO));
lastNum = 1;
count = 1;
for i=1:size(resultO,2),
    doVarName = result(resultO(i):resultM(i));
    objectIdeal(i) = str2num(result((resultN(i)+1)));
    objects = doVarName(4);
    if rand(1) > .5,
        fprintf('*******************************************************************\n');
        fprintf('* TRAINING %d\n',count);
        fprintf('*******************************************************************\n');
        command = ['../bin/test-NPclassify ../data_sets/num/',doVarName,' ',objects];
        trainList(i) = 1;
        unix(command);
        count = count + 1;
    else
        trainList(i) = 0;
    end
end
fprintf('\n');
load('train_set.dat');
%INS = train_set(:,2:3);
INP(:,1) = train_set(:,1); INPl(1,:) =                      'SPACE                    '; %space
INP(:,2) = train_set(:,2); INPl(2,:) =                      'DENSITY                  '; %density
INP(:,3) = train_set(:,3); INPl(3,:) =                      'OBJECTS                  '; %objects
INP(:,4) = INP(:,1) .* INP(:,2); INPl(4,:) =                'SPACE x DENSITY          ';
INP(:,5) = INP(:,1) .* INP(:,3); INPl(5,:) =                'SPACE x OBJECTS          ';
INP(:,6) = INP(:,2) .* INP(:,3); INPl(6,:) =                'DENSITY x OBJECTS        ';
INP(:,7) = INP(:,1) .* INP(:,2) .* INP(:,3); INPl(7,:) =    'SPACE x DENSITY x OBJECTS';
OUT(:,1) = train_set(:,4); %distanceCut
OUT(:,2) = train_set(:,5); %childCut

objectReal1 = train_set(:,6);
objectError1 = zeros(size(objectReal1));

count = 1;

for i=1:size(trainList,2),
    if trainList(i) == 1,
        objectError1(count) = abs(objectIdeal(i) - objectReal1(count));
        count = count + 1;
        %fprintf('count %d\n',count);
    end
end

meanError1 = mean(objectError1);
stdError1 = std(objectError1);

polyDem = 2;

for i=1:2
    fprintf('\n*************************************\n');
    for j=1:7
        Fit(:,i,j) = polyfit(INP(:,j),OUT(:,i),polyDem);
        Val(:,i,j) = polyval(Fit(:,i,j),INP(:,j));
        Err(:,i,j) = abs(Val(:,i,j) - OUT(:,i));
        stdErr(i,j) = std(Err(:,i,j));
        meanErr(i,j) = mean(Err(:,i,j));
        fprintf('(%d,%d) ERROR %f, STD %f\t%s\n',i,j,meanErr(i,j),stdErr(i,j),INPl(j,:));
    end
end


command = ['echo polyDensObjectCut1 ',num2str(Fit(1,1,2)),' > polySet.conf'];
unix(command,'-echo');
command = ['echo polyDensObjectCut2 ',num2str(Fit(2,1,2)),' >> polySet.conf'];
unix(command,'-echo');
command = ['echo polyDensObjectCut3 ',num2str(Fit(3,1,2)),' >> polySet.conf'];
unix(command,'-echo');
command = ['echo polySpaceChildCut1 ',num2str(Fit(1,2,1)),' >> polySet.conf'];
unix(command,'-echo');
command = ['echo polySpaceChildCut2 ',num2str(Fit(2,2,1)),' >> polySet.conf'];
unix(command,'-echo');
command = ['echo polySpaceChildCut3 ',num2str(Fit(3,2,1)),' >> polySet.conf'];
unix(command,'-echo');

count = 1;
unix('rm -f /lab/mundhenk/code/CODE_BASE_2/saliency/src/train_set.dat');
for i=1:size(trainList,2),
    doVarName = result(resultO(i):resultM(i));
    objects = doVarName(4);
    if trainList(i) == 0;
        fprintf('*******************************************************************\n');
        fprintf('* TESTING %d\n',count);
        fprintf('*******************************************************************\n');
        command = ['../bin/test-NPclassify ../data_sets/num/',doVarName,' ',objects];
        unix(command);
        count = count + 1;
    end
end
fprintf('\n');
load('train_set.dat');
objectReal2 = train_set(:,6);
objectError2 = zeros(size(objectReal1));

count = 1;

for i=1:size(trainList,2),
    if trainList(i) == 0,
        objectError2(count) = abs(objectIdeal(i) - objectReal2(count));
        count = count + 1;
    end
end

meanError2 = mean(objectError2);
stdError2 = std(objectError2);
fprintf('\n(PROLOG) NUMBER %d MEAN ERROR %f STD ERROR %f\n',count-1,meanError1,stdError1);
fprintf('\n(EPILOG) NUMBER %d MEAN ERROR %f STD ERROR %f\n',count-1,meanError2,stdError2);

%unix('mv -f ../data_sets/num/*.ppm ../data_sets/num.result');











