% trainNPclassify2 %
% This is a function class called by for instance a simplex which is used
% to improve the workings of the NPclassify algorithm

function f = trainNPclassify2(X)
%X = [2;5.949;2.3447;-0.77114;9.1723;27.6191;-0.0024981;1.415;0.74183];
%X =  [-0.1;5.949;0.68604;32.2476;-10.7899;69.0802;-0.0024981;1.415;0.74183];
%X =  [1.9707;5.9485;2.6479;4.1252;9.863;57.9016;0.048849;1.0329;0.80935];
%X = [2;5.8417;2.5702;2.0786;9.6807;28.8574;-0.0098149;1.6478;0.67709];
%baseChildIC = -0.20775;
%baseChild = 9.6807;
%baseSize = 29.1816;

baseChildIC = 3;
baseChild = 5;
baseSize = 100;

%basePoly1 = -0.051474;
%basePoly2 = 0.48111;
%basePoly3 = -0.19432;

% adjust sizing as needed
X(4) = ((X(4) - baseChildIC) * 10) + X(4);
X(5) = ((X(5) - baseChild) * 10) + X(5);
X(6) = ((X(6) - baseSize) * 100) + X(6);
%X(7) = ((X(7) - basePoly1) * -0.1) + X(7);
%X(8) = ((X(8) - basePoly2) * .1) + X(8);
%X(9) = ((X(9) - basePoly3) * -0.1) + X(9);
cd /lab/mundhenk/code/CODE_BASE_2/saliency/src;
unix('rm -f /lab/mundhenk/code/CODE_BASE_2/saliency/src/train_set.dat');
result = '';
resultComma = '';
doVarName = [];
doVarNameList = '';
command = '';
objects = [];
[status,result] = unix('ls -xm ../data_sets/num/');
resultO = findstr(result,'o');
resultM = findstr(result,'m');
resultN = findstr(result,'j');

fprintf('********************************************************\n');
fprintf('* NEW ROUND\n');
fprintf('********************************************************\n');
fprintf('* TRYING\n');
X
time = clock;
saveState = ['echo STATE ',date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
            ' ',num2str(X(1)),' ',num2str(X(2)),' ',num2str(X(3)),' ',...
            num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),' ' ,num2str(X(7)),' ',num2str(X(8)),' ',num2str(X(9)),...
            ' >> trainNPclassify2.log'];
unix(saveState);
for i=1:size(resultO,2),
    doVarName = result(resultO(i):resultM(i));
    objectIdeal(i) = str2num(result((resultN(i)+1)));
    objects = doVarName(4);
    fprintf('TRAINING %d\n',i);
    command = ['../bin/test-NPclassify ../data_sets/num/',doVarName,' ',objects,' ',num2str(X(1)),...
            ' ',num2str(X(2)),' ',num2str(X(3)),' ',num2str(X(4)),' ',num2str(X(5)),' ',num2str(X(6)),...
            ' ',num2str(X(7)),' ',num2str(X(8)),' ',num2str(X(9))];
    unix(command);
    %fprintf('.');

end
fprintf('\n');
load('train_set.dat');
%train_set(:,6)
%objectIdeal
ff = ((sum(abs(train_set(:,6) - transpose(objectIdeal))))/size(resultO,2)); %objects
CH = sum(objectIdeal)/size(resultO,2);
fprintf('ERROR %f\n',ff);
fprintf('CHANCE %f\n',CH);
time = clock;
saveState2 = ['echo ERROR ',date,' ',num2str(time(4)),':',num2str(time(5)),':',num2str(time(6)),...
            ' ', num2str(ff),' >> trainNPclassify2.log'];
unix(saveState2);
cd /lab/mundhenk/code/CODE_BASE_2/saliency/matlab;
f = ((sum(abs(train_set(:,6) - transpose(objectIdeal))))/size(resultO,2));
