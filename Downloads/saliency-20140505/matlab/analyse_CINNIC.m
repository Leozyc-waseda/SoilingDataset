cardinal = [];
cardLabel = [];
cardRatio = [];
cardLog = [];
load('0/finalMatlab.0.dat'); 
if size(X0_finalMatlab,2) > 0,
	Major0 = X0_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor0 = X0_finalMatlab(:,1);
   siz(1) = size(X0_finalMatlab,1); cardinal = cat(1,cardinal,Major0); cardLabel = cat(1,cardLabel,Minor0);
end
load('1/finalMatlab.1.dat'); 
if size(X1_finalMatlab,2) > 0,
   Major1 = X1_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor1 = X1_finalMatlab(:,1);
   siz(2) = size(X1_finalMatlab,1); cardinal = cat(1,cardinal,Major1); cardLabel = cat(1,cardLabel,Minor1);
end
load('2/finalMatlab.2.dat'); 
if size(X2_finalMatlab,2) > 0,
   Major2 = X2_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor2 = X2_finalMatlab(:,1);
   siz(3) = size(X2_finalMatlab,1); cardinal = cat(1,cardinal,Major2); cardLabel = cat(1,cardLabel,Minor2);
end
load('3/finalMatlab.3.dat'); 
if size(X3_finalMatlab,2) > 0,
	Major3 = X3_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor3 = X3_finalMatlab(:,1);
   siz(4) = size(X3_finalMatlab,1); cardinal = cat(1,cardinal,Major3); cardLabel = cat(1,cardLabel,Minor3);
end
load('4/finalMatlab.4.dat'); 
if size(X4_finalMatlab,2) > 0,
   Major4 = X4_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor4 = X4_finalMatlab(:,1);
   siz(5) = size(X4_finalMatlab,1); cardinal = cat(1,cardinal,Major4); cardLabel = cat(1,cardLabel,Minor4);
end
load('5/finalMatlab.5.dat'); 
if size(X5_finalMatlab,2) > 0,
   Major5 = X5_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor5 = X5_finalMatlab(:,1);
   siz(6) = size(X5_finalMatlab,1); cardinal = cat(1,cardinal,Major5); cardLabel = cat(1,cardLabel,Minor5);
end
load('6/finalMatlab.6.dat'); 
if size(X6_finalMatlab,2) > 0,
   Major6 = X6_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor6 = X6_finalMatlab(:,1);
   siz(7) = size(X6_finalMatlab,1); cardinal = cat(1,cardinal,Major6); cardLabel = cat(1,cardLabel,Minor6);
end
load('7/finalMatlab.7.dat'); 
if size(X7_finalMatlab,2) > 0,
   Major7 = X7_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor7 = X7_finalMatlab(:,1);
   siz(8) = size(X7_finalMatlab,1); cardinal = cat(1,cardinal,Major7); cardLabel = cat(1,cardLabel,Minor7);
end
load('8/finalMatlab.8.dat'); 
if size(X8_finalMatlab,2) > 0,
   Major8 = X8_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor8 = X8_finalMatlab(:,1);
   siz(9) = size(X8_finalMatlab,1); cardinal = cat(1,cardinal,Major8); cardLabel = cat(1,cardLabel,Minor8);
end
load('9/finalMatlab.9.dat'); 
if size(X9_finalMatlab,2) > 0,
	Major9 = X9_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor9 = X9_finalMatlab(:,1);
   siz(10) = size(X9_finalMatlab,1); cardinal = cat(1,cardinal,Major9); cardLabel = cat(1,cardLabel,Minor9);
end
load('10/finalMatlab.10.dat'); 
if size(X10_finalMatlab,2) > 0,
	Major10 = X10_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor10 = X10_finalMatlab(:,1);
   siz(11) = size(X10_finalMatlab,1); cardinal = cat(1,cardinal,Major10); cardLabel = cat(1,cardLabel,Minor10);
end
load('11/finalMatlab.11.dat'); 
if size(X11_finalMatlab,2) > 0,
	Major11 = X11_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor11 = X11_finalMatlab(:,1);
   siz(12) = size(X11_finalMatlab,1); cardinal = cat(1,cardinal,Major11); cardLabel = cat(1,cardLabel,Minor11);
end
load('12/finalMatlab.12.dat'); 
if size(X12_finalMatlab,2) > 0,
	Major12 = X12_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor12 = X12_finalMatlab(:,1);
   siz(13) = size(X12_finalMatlab,1); cardinal = cat(1,cardinal,Major12); cardLabel = cat(1,cardLabel,Minor12);
end
load('13/finalMatlab.13.dat'); 
if size(X13_finalMatlab,2) > 0,
   Major13 = X13_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor13 = X13_finalMatlab(:,1);
   siz(14) = size(X13_finalMatlab,1); cardinal = cat(1,cardinal,Major13); cardLabel = cat(1,cardLabel,Minor13);
end
load('14/finalMatlab.14.dat'); 
if size(X14_finalMatlab,2) > 0,
	Major14 = X14_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor14 = X14_finalMatlab(:,1);
   siz(15) = size(X14_finalMatlab,1); cardinal = cat(1,cardinal,Major14); cardLabel = cat(1,cardLabel,Minor14);
end
load('15/finalMatlab.15.dat'); 
if size(X15_finalMatlab,2) > 0,
	Major15 = X15_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor15 = X15_finalMatlab(:,1);
   siz(16) = size(X15_finalMatlab,1); cardinal = cat(1,cardinal,Major15); cardLabel = cat(1,cardLabel,Minor15);
end
load('16/finalMatlab.16.dat'); 
if size(X16_finalMatlab,2) > 0,
	Major16 = X16_finalMatlab(:,2:size(X0_finalMatlab,2)); Minor16 = X16_finalMatlab(:,1);
   siz(17) = size(X16_finalMatlab,1); cardinal = cat(1,cardinal,Major16); cardLabel = cat(1,cardLabel,Minor16);   
end

load('rawData1.dat');
load('rawData2.dat');

rawMult1 = zeros(size(cardinal,2));
rawMult1(2,2) = 1; rawMult1(4,4) = 1; rawMult1(6,6) = 1; rawMult1(8,8) = 1; rawMult1(10,10) = 1; rawMult1(14,14) = 1; rawMult1(15,15) = 1;
rawMult2 = rawMult1;
rawMult1(15,15) = 0;

for i=1:size(cardinal,1),
   for j=2:size(cardinal,2),
      cardRatio(i,j) = cardinal(i,j)/cardinal(i,1);
      cardLog(i,j) = log10(cardRatio(i,j));
   end
end

rawError1 = (cardLog * transpose(rawMult1));
rawError2 = (cardLog * transpose(rawMult2));
totalError1 = zeros(size(cardinal,1),1);
totalError2 = zeros(size(cardinal,1),1);
for i=1:size(cardinal,1),
   for j=2:size(cardinal,2),
      if rawError1(i,j) ~= 0,
      	rawError1(i,j) = abs(rawError1(i,j) - rawData1(i));
         totalError1(i) = totalError1(i) + rawError1(i,j);
      end
      if rawError2(i,j) ~= 0,
      	rawError2(i,j) = abs(rawError2(i,j) - rawData2(i));
         totalError2(i) = totalError2(i) + rawError2(i,j);
      end
   end
   meanError1(i) = totalError1(i)/6;
   meanError2(i) = totalError2(i)/7;
end

std1 = zeros(size(cardinal,1),1);
std2 = zeros(size(cardinal,1),1);

for i=1:size(cardinal,1),
   for j=2:size(cardinal,2),
      if rawError1(i,j) ~= 0,
         std1(i) = std1(i) + (meanError1(i) - rawError1(i,j))^2;
      end   
		if rawError2(i,j) ~= 0,
     		std2(i) = std2(i) + (meanError2(i) - rawError2(i,j))^2;
   	end
   end
   plot(transpose(rawError1(i)));
   plot(transpose(rawError2(i)));
   std1(i) = sqrt(std1(i))/6;
   std2(i) = sqrt(std2(i))/7;
end

error = cat(1,cardLog,rawData1);


hold on
P1 = stem(transpose(rawData2),'bd');
P1a = stem(transpose(rawData1),'rd');
P2 = plot(transpose(cardLog));
hold off

P3 = [P1(1);P1a(1);P2];
legend(P3,'PS1','PS2','1','2','3','4','5');
xlabel('Lambda Seperation')
ylabel('Enhancement (log units)')

title('log error Polat Sagi V. CINNIC');

cardinal
cardLabel
cardRatio
rawData1
rawData2
cardLog
rawError1
rawError2
totalError1
totalError2
meanError1
meanError2
std1
std2