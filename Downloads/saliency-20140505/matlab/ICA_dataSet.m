% find ICA over a total data set
% filename format: <Set>.<SetSeries>.<ID>.dat
% set to load
Set = 'CRW';
% set series to load
SetSeries = 'nat';

%X = load(['../features.',SetSeries,'.dat']);

% how many features per channel
featureStep = 6;

% run PCA on the data if you like
%fprintf('Doing PCA\n');
%[PX,transMat] = prepca(X',0.01);

fprintf('Doing ICA\n');

AF = [];
WF = [];
n = 1;

% run ICA per channel on all the data to start with
for j=1:featureStep:48;  
   if (sum(sum(X(:,j:(j+featureStep-1)),2),1)) ~= 0,
		fprintf('\nICA on featues %d to %d\n',j,j+featureStep-1);
      [XM,A,W] = fastica(X(:,j:(j+featureStep-1))','lastEig',2);
      figure(n+1) 
      subplot(2,1,1)
      plot(XM(:,1));
      subplot(2,1,2)
      plot(XM(:,2));
      title('ICA OUTPUT');
   	AF = cat(1,AF,A);
      WF = cat(1,WF,W);
      ICA(n).A = A;
      ICA(n).W = W;
      logDat(1,n) = j;
      logDat(2,n) = (j+featureStep-1);
      n = n + 1;
   else
      fprintf('SKIPING zero data\n');
   end   
end
pwd
% unmix all data sets using the unmixing matrix from ICA
fprintf('Taking unmixing matrix over sample files\n');
doDir = ['*.300.*.dat']
D = dir(doDir);
n = 1;
FXout = [];
for i=1:size(D,1),
   XI = load(D(i).name);
   fprintf('loading %s\n',D(i).name);
   FXout = [];
   for j=1:featureStep:48;  
      if (sum(sum(XI(:,j:(j+featureStep-1)),2),1)) ~= 0,
         Xout = XI(:,j:(j+featureStep-1)) * ICA(n).W';
         FXout = cat(2,FXout,Xout);
      end
   end
   outname = ['ICA.'];
   writeMe = cat(2,outname,D(i).name);
   fprintf('Saving %s\n',writeMe);
   fid = fopen(writeMe,'w');
   for j=1:size(FXout,1),
      for k=1:size(FXout,2),
      	fprintf(fid,'%f\t',FXout(j,k));
		end    
      fprintf(fid,'\n');
   end   
   fclose(fid);
end

         
fprintf('Saving Data\n');
%save PCA.dat transMat
save ICAmix.dat AF
save ICAsep.dat WF
save log.dat logDat
fprintf('Done\n');